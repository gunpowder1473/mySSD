import tensorflow as tf
import math
from Common import common_methods
from tensorflow.python.framework import tensor_shape

slim = tf.contrib.slim


def ssdDefaultboxResult(inputs, num_classes, sizes, ratios=[1], normalization=-1):
    net = inputs
    if normalization > 0:
        net = common_methods.l2_normalization(net, scaling=True)
    # defaultbox个数，每个featuremap除去对应的ratios的以外，额外添加2个，其aspect ratios = 1
    # 大小为sk与sqrt（sk*sk+1）.
    num_default = len(sizes) + len(ratios)
    # 分别用3*3的卷积核去预测location与class的输出
    # Location. (x,y,w,h) 4个offset
    num_loc_pred = num_default * 4
    loc_pred = slim.conv2d(net, num_loc_pred, [3, 3], activation_fn=None, scope='conv_loc')
    # loc_pred转为NHWC
    loc_pred = common_methods.dataFormatChange(loc_pred)
    # 相当于增添了一维，将第三维拆开为num_default*4
    loc_pred = tf.reshape(loc_pred, common_methods.tensorShape(loc_pred, 4)[:-1] + [num_default, 4])
    # Class. 30种
    num_cls_pred = num_default * num_classes
    # 卷积核：3*3*net的第三维，无激活函数，30*num_default个卷积核，输出前两维同net，第三维30*num_default
    # 得到了每个defaultbox对30类的预测输出
    cls_pred = slim.conv2d(net, num_cls_pred, [3, 3], activation_fn=None, scope='conv_cls')
    # cls_pred转为NHWC
    cls_pred = common_methods.dataFormatChange(cls_pred)
    # 相当于增添了一维，将第三维拆开为num_default*30
    # 得到了每个defaultbox对4个误差的预测输出
    cls_pred = tf.reshape(cls_pred, common_methods.tensorShape(cls_pred, 4)[:-1] + [num_default, num_classes])
    return cls_pred, loc_pred


def ssd_losses(logits, localisations, gclasses, glocalisations, gscores, match_threshold=0.5,
               negative_ratio=3., alpha=1., scope=None):
    with tf.name_scope(scope, 'ssd_losses'):
        # 传到这里时每一个defaultbox对应的class是得分最高的
        # logits对应6个卷积层的输出，每一层的shape为（batch, x, x, defaultbox_per_featuremap, num_classes)
        # 其中x是对应的卷积层的大小
        lshape = common_methods.tensorShape(logits[0], 5)
        num_classes = lshape[-1]
        batch_size = lshape[0]
        flogits = []
        fgclasses = []
        fgscores = []
        flocalisations = []
        fglocalisations = []
        for i in range(len(logits)):
            flogits.append(tf.reshape(logits[i], [-1, num_classes]))
            fgclasses.append(tf.reshape(gclasses[i], [-1]))
            fgscores.append(tf.reshape(gscores[i], [-1]))
            flocalisations.append(tf.reshape(localisations[i], [-1, 4]))
            fglocalisations.append(tf.reshape(glocalisations[i], [-1, 4]))
        # num_defaultbox * num_classes,这里num_defaultbox是6个卷积层内包含的defaultbox的总和
        logits = tf.concat(flogits, axis=0)
        # num_defaultbox
        gclasses = tf.concat(fgclasses, axis=0)
        # num_defaultbox
        gscores = tf.concat(fgscores, axis=0)
        # num_defaultbox * 4
        localisations = tf.concat(flocalisations, axis=0)
        # num_dfaultbox * 4
        glocalisations = tf.concat(fglocalisations, axis=0)
        # glabels = tf.reshape(glabels, [-1])
        # glabels = tf.cast(glabels, tf.int32)
        # num_labels = glabels._shape_as_list()[0]
        dtype = logits.dtype
        # 去掉小于阈值的
        pmask = gscores > match_threshold
        # temp_mask = -(tf.cast(pmask, tf.int64))
        # # 超过阈值中被选中的类
        # be_choosed = temp_mask * gclasses
        # be_choosed = tf.cast(be_choosed, tf.int32)
        # in_or_out_flag = []
        # i = 0
        #
        # def isIn():
        #     return True
        #
        # def isOut():
        #     return False
        #
        # def shouldEnd(i):
        #     return tf.less(i, num_classes)
        #
        # def shouldLoop(judge_result, i, j):
        #     result = tf.logical_and(judge_result, tf.less(j, 1))
        #     return result
        #
        # def isInLabels(i):
        #     find = tf.convert_to_tensor(False)
        #     for j in range(num_labels):
        #         result = tf.cond(tf.equal(i, glabels[j]), isIn, isOut)
        #         result = tf.cast(result, tf.bool)
        #         find = tf.logical_or(result, find)
        #     return find
        #
        # def loopBase(judge_result, i, j):
        #     # 判断被选中的类中有无该类
        #     temp_equal = tf.equal(i - 1, be_choosed)
        #     temp_change = tf.cast(temp_equal, tf.int32)
        #     sum = tf.reduce_sum(temp_change)
        #     # 和大于0，说明被选中的类中存在该类
        #     judge = tf.greater(sum, 0)
        #     # (类，是否需要补（即不存在）)
        #     in_or_out_flag.append((i, judge))
        #     j += 1
        #     return judge_result, i, j
        #
        # def loopBody(i):
        #     # 首先判断该类有没有被标注在当前图片中
        #     # 若在，则执行loopBase
        #     judge_result = isInLabels(i)
        #     j = 0
        #     i += 1
        #     judge_result, i, j = tf.while_loop(shouldLoop, loopBase, [judge_result, i, j])
        #     return i
        #
        # i = tf.while_loop(shouldEnd, loopBody, [i])
        #
        # def needFind(judge_tuple, j):
        #     result = tf.greater(1, 2)#tf.logical_and(judge_tuple[1], tf.less(j, 1))
        #     return result
        #
        # def addMax(judge_tuple, j):
        #     # 将所有scores从大到小排列
        #     # gscores与gclasses下标统一，通过下标去找到分类
        #     _, val_classes = tf.nn.top_k(gscores, k=gscores._shape_as_list()[0])
        #     not_get = tf.convert_to_tensor(True)
        #     tf.while_loop(startFind, findLoop, [not_get, i, val_classes, judge_tuple])
        #     # 只执行一次
        #     j += 1
        #     return judge_result, j
        #
        # def startFind(not_get, i, val_classes, judge_tuple):
        #     result = tf.logical_and(not_get, tf.less(i, gclasses._shape_as_list()[0]))
        #     return result
        #
        # def findLoop(not_get, i, val_classes, judge_tuple):
        #     # i是遍历索引，val_classes[i]为在排序之前的索引，gclasses[val_classes[i]]为对应的分类
        #     value = tf.cast(judge_tuple[0], tf.int32)
        #     mask = tf.equal(gclasses[val_classes[i]], value)
        #     # mask为真说明找到了，可以跳出循环
        #     not_get = tf.where(mask, False, True)
        #     i += 1
        #     return not_get, i, val_classes, judge_tuple
        #
        # for i in range(len(in_or_out_flag)):
        #     j = 0
        #     judge_tuple = in_or_out_flag[i]
        #     tf.while_loop(needFind, addMax, [judge_tuple, j])

        # 计算所有超过阈值的bbox的总数

        # 得到一个包含所有本次图像中包含的bbox的类别的无重复的列表
        actual_all_classes, _ = tf.unique(gclasses)
        actual_all_classes = tf.reshape(actual_all_classes, [-1])

        # 得到一个包含所有本次图像中大于阈值的bbox的类别的无重复的列表
        # 首先留下被选中的bboxes的类别
        choosen_class = tf.boolean_mask(gclasses, pmask)
        actual_choosen_classes, _ = tf.unique(choosen_class)
        actual_choosen_classes = tf.reshape(actual_choosen_classes, [-1])

        def notFind(i, pmask, include_not_choosen_num, include_not_choosen):
            mask = tf.less(i, include_not_choosen_num)
            return mask

        def findLoop(i, pmask, include_not_choosen_num, include_not_choosen):
            mask = tf.equal(gclasses, include_not_choosen[i])
            fmask = tf.cast(mask, gscores.dtype)
            # 留下那些得分都不过0.5的defaultbboxes
            cur_class_scores = gscores * fmask
            max_scores_index = tf.argmax(cur_class_scores, dimension=0)
            temp_pmask = tf.sparse_to_dense(max_scores_index, pmask.get_shape(), tf.constant(True), tf.constant(False))
            pmask = tf.logical_or(temp_pmask, pmask)
            return i + 1, pmask, include_not_choosen_num, include_not_choosen

        def needAdd(pmask):
            # include_not_choosen = tf.slice(tf.expand_dims(actual_all_classes, 0),
            #                                 [0, tf.shape(actual_choosen_classes)[0]],
            #                                 [1, tf.shape(actual_all_classes)[0] - tf.shape(actual_choosen_classes)[0]])
            # include_not_choosen = include_not_choosen[0]
            include_not_choosen = tf.boolean_mask(gclasses, tf.logical_not(pmask))
            include_not_choosen_num = tf.shape(include_not_choosen)[0]
            j = 0
            _, pmask, _, _ = tf.while_loop(notFind, findLoop,
                                           [j, pmask, include_not_choosen_num, include_not_choosen])
            return pmask

        # 长度不一样，说明有类别对应的bboxes得分均不过0.5
        is_need_add = tf.equal(tf.shape(actual_all_classes)[0], tf.shape(actual_choosen_classes)[0])
        pmask = tf.cond(is_need_add, lambda: needAdd(pmask), lambda: pmask)

        fpmask = tf.cast(pmask, dtype)
        n_positives = tf.reduce_sum(fpmask)

        # 经过前面的筛选后，去掉的negativebox很多，远多于positivebox
        # 使得两者不均衡，在置信度损失中第一项的x因子的大部分都是0，损失函数很小
        # 难以收敛，所以在negativebox中选择较大的，使得最后两者比例在3:1（这里由negative ratio决定）
        no_classes = tf.cast(pmask, tf.int32)
        predictions = slim.softmax(logits)
        nmask = tf.logical_and(tf.logical_not(pmask), gscores > -0.5)
        fnmask = tf.cast(nmask, dtype)
        # predictions[:,0]中0对应于Lconf中Neg的c的上表0，即第0个分类（背景,即无匹配对象）
        nvalues = tf.where(nmask, predictions[:, 0], 1. - fnmask)
        nvalues_flat = tf.reshape(nvalues, [-1])
        max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
        n_neg = tf.cast(negative_ratio * n_positives, tf.int32) + batch_size
        n_neg = tf.minimum(n_neg, max_neg_entries)
        # 对无匹配类置信度最小，相当于对其他类置信度最大
        val, _ = tf.nn.top_k(-nvalues_flat, k=n_neg)
        # 选择其中最大的n_neg个
        max_neg_val = -val[-1]
        nmask = tf.logical_and(nmask, nvalues < max_neg_val)
        fnmask = tf.cast(nmask, dtype)

        # 交叉熵
        with tf.name_scope('cross_entropy_pos'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=gclasses)
            loss = tf.div(tf.reduce_sum(loss * fpmask), batch_size, name='value')
            tf.losses.add_loss(loss)

        with tf.name_scope('cross_entropy_neg'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=no_classes)
            loss = tf.div(tf.reduce_sum(loss * fnmask), batch_size, name='value')
            tf.losses.add_loss(loss)

        # Loc损失
        with tf.name_scope('localization'):
            weights = tf.expand_dims(alpha * fpmask, axis=-1)
            loss = common_methods.abs_smooth(localisations - glocalisations)
            loss = tf.div(tf.reduce_sum(loss * weights), batch_size, name='value')
            tf.losses.add_loss(loss)
