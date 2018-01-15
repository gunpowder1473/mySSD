import tensorflow as tf
import numpy as np
from Common import common_methods


def bboxesResize(bbox_ref, bboxes, name=None):
    with tf.name_scope(name, 'bboxes_resize'):
        v = tf.stack([bbox_ref[0], bbox_ref[1], bbox_ref[0], bbox_ref[1]])
        bboxes = bboxes - v
        s = tf.stack([bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1],
                      bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1]])
        bboxes = bboxes / s
        return bboxes


def bboxesFilter(labels, bboxes, threshold=0.7, scope=None, isFlip=None):
    with tf.name_scope(scope, 'bboxes_filter', [labels, bboxes]):
        scores = bboxesFraction(tf.constant([0, 0, 1, 1], bboxes.dtype), bboxes, label=labels, isFlip=isFlip)
        mask = scores > threshold
        # 官方示例：
        # 2-D example
        # tensor = [[1, 2], [3, 4], [5, 6]]
        # mask = np.array([True, False, True])
        # boolean_mask(tensor, mask)  # [[1, 2], [5, 6]]
        labels = tf.boolean_mask(labels, mask)
        bboxes = tf.boolean_mask(bboxes, mask)
        bboxes = tf.where(tf.greater(bboxes, 0), bboxes, tf.zeros_like(bboxes))
        bboxes = tf.where(tf.greater(bboxes, 1), tf.ones_like(bboxes), bboxes)
        return labels, bboxes


def bboxesFraction(bbox_ref, bboxes, name=None, label=None, isFlip=None):
    with tf.name_scope(name, 'BboxesFraction'):
        # 转置
        bboxes = tf.transpose(bboxes)
        bbox_ref = tf.transpose(bbox_ref)
        int_ymin = tf.maximum(bboxes[0], bbox_ref[0])
        int_xmin = tf.maximum(bboxes[1], bbox_ref[1])
        int_ymax = tf.minimum(bboxes[2], bbox_ref[2])
        int_xmax = tf.minimum(bboxes[3], bbox_ref[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        if label is not None:
            # mask0 = tf.equal(label, 1)
            # mask1 = tf.equal(label, 5)
            # mask2 = tf.equal(label, 6)
            # mask3 = tf.equal(label, 7)
            # mask4 = tf.equal(label, 8)
            # all_mask = tf.logical_or(mask0, mask1)
            # all_mask = tf.logical_or(mask1, all_mask)
            # all_mask = tf.logical_or(mask3, all_mask)
            # all_mask = tf.logical_or(mask4, all_mask)
            all_mask = tf.equal(label, 2)
            # 重叠交集
            inter_vol = tf.where(all_mask, h * w * 0.7 / 0.8, h * w)
            if isFlip is not None:
                # 发生Flip就取消文字相关的两个类
                mask2 = tf.equal(label, 2)
                mask5 = tf.logical_and(mask2, isFlip)
                # mask6 = tf.logical_and(mask4, isFlip)
                # mask7 = tf.logical_or(mask5, mask6)
                # temp = 1. - tf.cast(mask7, inter_vol.dtype)
                temp = 1. - tf.cast(mask5, inter_vol.dtype)
                inter_vol = inter_vol * temp
        else:
            inter_vol = h * w
        # bbox全集
        bboxes_vol = (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1])
        scores = tf.where(tf.greater(bboxes_vol, 0), tf.div(inter_vol, bboxes_vol), tf.zeros_like(inter_vol),
                          'intersection')
        return scores


def bboxesFlipLR(bboxes):
    bboxes = tf.stack([bboxes[:, 0], 1 - bboxes[:, 3], bboxes[:, 2], 1 - bboxes[:, 1]], axis=-1)
    return bboxes


def bboxesFlipUD(bboxes):
    bboxes = tf.stack([1 - bboxes[:, 2], bboxes[:, 1], 1 - bboxes[:, 0], bboxes[:, 3]], axis=-1)
    return bboxes


def bboxesEncodeOneLayer(labels, bboxes, defaultboxes, num_classes, prior_scaling=[0.1, 0.1, 0.2, 0.2]):
    # x,y是default box的中心，h,w为高和宽
    yref, xref, href, wref = defaultboxes
    ymin = yref - href / 2.
    xmin = xref - wref / 2.
    ymax = yref + href / 2.
    xmax = xref + wref / 2.

    # default box 体积
    vol_defaults = (xmax - xmin) * (ymax - ymin)
    shape = (yref.shape[0], yref.shape[1], href.size)
    feat_labels = tf.zeros(shape, dtype=tf.int64)
    feat_scores = tf.zeros(shape, dtype=tf.float32)

    feat_ymin = tf.zeros(shape, dtype=tf.float32)
    feat_xmin = tf.zeros(shape, dtype=tf.float32)
    feat_ymax = tf.ones(shape, dtype=tf.float32)
    feat_xmax = tf.ones(shape, dtype=tf.float32)

    def calJaccard(bbox):
        int_ymin = tf.maximum(ymin, bbox[0])
        int_xmin = tf.maximum(xmin, bbox[1])
        int_ymax = tf.minimum(ymax, bbox[2])
        int_xmax = tf.minimum(xmax, bbox[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # bound box 与 default box 交集
        inter_vol = h * w
        # bound box 与 default box 并集
        union_vol = vol_defaults - inter_vol \
                    + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        # jaccard系数 = 交集/并集
        jaccard = tf.where(tf.greater(union_vol, 0), tf.div(inter_vol, union_vol), tf.zeros_like(inter_vol))
        return jaccard

    def isCheckAllClass(i, feat_labels, feat_scores, feat_ymin, feat_xmin, feat_ymax, feat_xmax):
        # 返回是否小于labels.shape[0],这里用于判断是否遍历所有类别
        r = tf.less(i, tf.shape(labels))
        return r[0]

    def bboxesMatch(i, feat_labels, feat_scores, feat_ymin, feat_xmin, feat_ymax, feat_xmax):
        label = labels[i]
        bbox = bboxes[i]
        jaccard = calJaccard(bbox)
        # mask = jacard 系数 > feat_scores ? True : False
        mask = tf.greater(jaccard, feat_scores)
        mask = tf.logical_and(mask, feat_scores > -0.5)
        # mask &= (label < num_classes),这里产生while-loop终止条件，即对所有的种类进行遍历
        mask = tf.logical_and(mask, label < num_classes)
        # mask = jaccard > feat_scorse ? True : False
        # bool -> 1,0/1.,0.
        imask = tf.cast(mask, tf.int64)
        fmask = tf.cast(mask, tf.float32)
        # feat_labels即表示jaccard系数最大的类
        feat_labels = imask * label + (1 - imask) * feat_labels
        # feat_scores = mask ? jaccard : feat_scores
        # jaccard : jaccard > feat_scores  && label < num_classes
        # 不变 : 其他
        feat_scores = tf.where(mask, jaccard, feat_scores)
        # 综上： jaccard > feat_scores feat_score被更新为jaccard，否则不变，亦即在满足 > -0.5的前提下，feat_scores保留最大值
        # fmask 为真则feat_box置为bbox，否则不变
        feat_ymin = fmask * bbox[0] + (1 - fmask) * feat_ymin
        feat_xmin = fmask * bbox[1] + (1 - fmask) * feat_xmin
        feat_ymax = fmask * bbox[2] + (1 - fmask) * feat_ymax
        feat_xmax = fmask * bbox[3] + (1 - fmask) * feat_xmax

        return [i + 1, feat_labels, feat_scores, feat_ymin, feat_xmin, feat_ymax, feat_xmax]

    i = 0
    # while(condition()){ body() }
    [i, feat_labels, feat_scores,
     feat_ymin, feat_xmin,
     feat_ymax, feat_xmax] = tf.while_loop(isCheckAllClass, bboxesMatch, [i, feat_labels, feat_scores,
                                                                          feat_ymin, feat_xmin, feat_ymax, feat_xmax])
    # feat_box 中心坐标
    feat_cy = (feat_ymax + feat_ymin) / 2.
    feat_cx = (feat_xmax + feat_xmin) / 2.
    feat_h = feat_ymax - feat_ymin
    feat_w = feat_xmax - feat_xmin

    # 下列计算对应于论文中置信损失计算 feat_c* -> g*  *ref -> d*  附加一个缩放比
    feat_cy = (feat_cy - yref) / href / prior_scaling[0]
    feat_cx = (feat_cx - xref) / wref / prior_scaling[1]
    feat_h = tf.log(feat_h / href) / prior_scaling[2]
    feat_w = tf.log(feat_w / wref) / prior_scaling[3]

    # 扩为4维，4维分别对于feat_cx,feat_cy,feat_w,feat_h
    feat_localizations = tf.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1)
    return feat_labels, feat_localizations, feat_scores


def bboxesEncodeAllLayer(labels, bboxes, defaultboxes, num_classes, prior_scaling=[0.1, 0.1, 0.2, 0.2],
                         scope='ssd_bboxes_encode'):
    with tf.name_scope(scope):
        target_labels = []
        target_localizations = []
        target_scores = []
        for i, layer in enumerate(defaultboxes):
            with tf.name_scope('bboxes_encode_block_%i' % i):
                t_labels, t_loc, t_scores = \
                    bboxesEncodeOneLayer(labels, bboxes, layer, num_classes, prior_scaling)
                target_labels.append(t_labels)
                target_localizations.append(t_loc)
                target_scores.append(t_scores)
        return target_labels, target_localizations, target_scores


def bboxDecodeOneLayer(feat_localizations, defaults_layer, prior_scaling=[0.1, 0.1, 0.2, 0.2]):
    yref, xref, href, wref = defaults_layer
    cx = feat_localizations[:, :, :, :, 0] * wref * prior_scaling[0] + xref
    cy = feat_localizations[:, :, :, :, 1] * href * prior_scaling[1] + yref
    w = wref * tf.exp(feat_localizations[:, :, :, :, 2] * prior_scaling[2])
    h = href * tf.exp(feat_localizations[:, :, :, :, 3] * prior_scaling[3])
    ymin = cy - h / 2.
    xmin = cx - w / 2.
    ymax = cy + h / 2.
    xmax = cx + w / 2.
    bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
    return bboxes


def bboxDecodeAllLayers(feat_localizations, defaults_layers, prior_scaling=[0.1, 0.1, 0.2, 0.2],
                        scope='ssd_bboxes_decode'):
    with tf.name_scope(scope):
        bboxes = []
        for i, defaults_layer in enumerate(defaults_layers):
            bboxes.append(bboxDecodeOneLayer(feat_localizations[i], defaults_layer, prior_scaling))
        return bboxes


# 相同的操作用numpy实现

def npBboxDecodeOneLayer(feat_localizations, defaults_layer, prior_scaling=[0.1, 0.1, 0.2, 0.2]):
    l_shape = feat_localizations.shape
    # (batch*x*x)*defaultbox_per_featuremap*4
    feat_localizations = np.reshape(feat_localizations, (-1, l_shape[-2], l_shape[-1]))
    yref, xref, href, wref = defaults_layer
    xref = np.reshape(xref, [-1, 1])
    yref = np.reshape(yref, [-1, 1])
    # (batch*x*x)*defaultbox_per_featuremap
    cx = feat_localizations[:, :, 0] * wref * prior_scaling[0] + xref
    cy = feat_localizations[:, :, 1] * href * prior_scaling[1] + yref
    w = wref * np.exp(feat_localizations[:, :, 2] * prior_scaling[2])
    h = href * np.exp(feat_localizations[:, :, 3] * prior_scaling[3])
    bboxes = np.zeros_like(feat_localizations)
    bboxes[:, :, 0] = cy - h / 2.
    bboxes[:, :, 1] = cx - w / 2.
    bboxes[:, :, 2] = cy + h / 2.
    bboxes[:, :, 3] = cx + w / 2.
    # batch*x*x*defaultbox_per_featuremap*4
    bboxes = np.reshape(bboxes, l_shape)
    return bboxes


def npBboxesSelectOneLayer(predictions_layer, localizations_layer, defaults_layer, select_threshold=0.5):
    # batch*x*x*defaultbox_per_featuremap*4
    localizations_layer = npBboxDecodeOneLayer(localizations_layer, defaults_layer)
    # batch*n*n*defaultbox_per_featuremap*num_classes
    p_shape = predictions_layer.shape
    # 1
    batch_size = p_shape[0] if len(p_shape) == 5 else 1
    # batch*(x*x*defaultbox_per_featuremap)*num_classes
    predictions_layer = np.reshape(predictions_layer, (batch_size, -1, p_shape[-1]))
    l_shape = localizations_layer.shape
    # batch*(x*x*defaultbox_per_featuremap)*4
    localizations_layer = np.reshape(localizations_layer, (batch_size, -1, l_shape[-1]))
    # 去掉分类0，即背景
    sub_predictions = predictions_layer[:, :, 1:]
    # 大于阈值的index
    indexs = np.where(sub_predictions > select_threshold)
    # 加1取得原来的分类
    classes = indexs[-1] + 1
    scores = sub_predictions[indexs]
    bboxes = localizations_layer[indexs[:-1]]

    return classes, scores, bboxes


def npBboxesSelectALLLayers(predictions_net, localizations_net, defaults_layer, select_threshold=0.5):
    l_classes = []
    l_scores = []
    l_bboxes = []
    for i in range(len(predictions_net)):  # 6
        classes, scores, bboxes = npBboxesSelectOneLayer(predictions_net[i], localizations_net[i], defaults_layer[i],
                                                         select_threshold)
        l_classes.append(classes)
        l_scores.append(scores)
        l_bboxes.append(bboxes)
    # 连接成一维数组
    classes = np.concatenate(l_classes, 0)
    scores = np.concatenate(l_scores, 0)
    bboxes = np.concatenate(l_bboxes, 0)
    return classes, scores, bboxes


def npBboxesSorted(classes, scores, bboxes, top_k=400):
    index = np.argsort(-scores)
    # 保留前top_k个最大的
    classes = classes[index][:top_k]
    scores = scores[index][:top_k]
    bboxes = bboxes[index][:top_k]
    return classes, scores, bboxes


def npBboxesLimit(bbox_ref, bboxes):
    # bboxes = np.copy(bboxes)
    bboxes = np.transpose(bboxes)
    bbox_ref = np.transpose(bbox_ref)
    bboxes[0] = np.maximum(bboxes[0], bbox_ref[0])
    bboxes[1] = np.maximum(bboxes[1], bbox_ref[1])
    bboxes[2] = np.minimum(bboxes[2], bbox_ref[2])
    bboxes[3] = np.minimum(bboxes[3], bbox_ref[3])
    bboxes = np.transpose(bboxes)
    return bboxes


def npBboxesResize(bbox_ref, bboxes):
    # bboxes = np.copy(bboxes)
    bboxes[:, 0] -= bbox_ref[0]
    bboxes[:, 1] -= bbox_ref[1]
    bboxes[:, 2] -= bbox_ref[0]
    bboxes[:, 3] -= bbox_ref[1]
    new_bboxes_hw = [bbox_ref[2] - bbox_ref[0], bbox_ref[3] - bbox_ref[1]]
    bboxes[:, 0] /= new_bboxes_hw[0]
    bboxes[:, 1] /= new_bboxes_hw[1]
    bboxes[:, 2] /= new_bboxes_hw[0]
    bboxes[:, 3] /= new_bboxes_hw[1]
    return bboxes


def npBboxesJaccard(bboxes1, bboxes2):
    bboxes1 = np.transpose(bboxes1)
    bboxes2 = np.transpose(bboxes2)

    int_ymin = np.maximum(bboxes1[0], bboxes2[0])
    int_xmin = np.maximum(bboxes1[1], bboxes2[1])
    int_ymax = np.minimum(bboxes1[2], bboxes2[2])
    int_xmax = np.minimum(bboxes1[3], bboxes2[3])

    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)
    int_vol = int_h * int_w

    vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])
    vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])
    jaccard = int_vol / (vol1 + vol2 - int_vol)
    return jaccard


def npBboxesFraction(bboxes_ref, bboxes2):
    bboxes_ref = np.transpose(bboxes_ref)
    bboxes2 = np.transpose(bboxes2)

    int_ymin = np.maximum(bboxes_ref[0], bboxes2[0])
    int_xmin = np.maximum(bboxes_ref[1], bboxes2[1])
    int_ymax = np.minimum(bboxes_ref[2], bboxes2[2])
    int_xmax = np.minimum(bboxes_ref[3], bboxes2[3])

    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)
    int_vol = int_h * int_w

    vol = (bboxes_ref[2] - bboxes_ref[0]) * (bboxes_ref[3] - bboxes_ref[1])
    score = int_vol / vol
    return score


def npBboxesNms(classes, scores, bboxes, nms_threshold=.45):
    should_keep = np.ones(scores.shape, dtype=np.bool)
    for i in range(scores.size - 1):
        if should_keep[i]:
            overlap = npBboxesJaccard(bboxes[i], bboxes[(i + 1):])
            # 非极大值抑制
            keep_overlap = np.logical_or(overlap < nms_threshold, classes[(i + 1):] != classes[i])
            should_keep[(i + 1):] = np.logical_and(should_keep[(i + 1):], keep_overlap)

    index = np.where(should_keep)
    return classes[index], scores[index], bboxes[index]


def npBboxesAccuracy(bbox, bboxes, classone, classes, threshold=.5):
    # 算出的bbox与实际bbox应该只有一个种类一致且重合度高的
    jaccard = npBboxesJaccard(bbox, bboxes)
    index = np.where(jaccard > threshold)[0]
    if len(index) == 0:
        return False
    # 留下重合多且分类一致的
    judge = np.equal(classone, classes[index])
    if np.count_nonzero(judge) != 1 or len(judge) == 0:
        return False
    else:
        return judge[0]
