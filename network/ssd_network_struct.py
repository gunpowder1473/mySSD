import tensorflow as tf
import numpy as np
import math
from Common import common_methods
from network import ssd_network_calc

slim = tf.contrib.slim


def SSDNet(inputs, num_classes, feat_layers, default_sizes, default_ratios, normalizations, is_training=True,
           dropout_keep_prob=0.5, prediction_fn=slim.softmax, reuse=None, scope='ssd_300'):
    end_points = {}
    with tf.variable_scope(scope, 'ssd_300', [inputs], reuse=reuse):
        # VGG16前5组，至conv5_3
        # 卷积核：3*3*3，conv1_1,conv1_2，64个卷积核，300+2-3+1=300，输出300*300*64
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        end_points['block1'] = net
        # 最大池化：2*2内取最大值，步长为2，300/2=150，输出150*150*64
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        # 卷积核：3*3*64，conv2_1,conv2_2，128个卷积核，300+2-3+1=300，输出300*300*128
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        end_points['block2'] = net
        # 最大池化：2*2内取最大值，步长为2，150/2=150，输出75*75*128
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        # 卷积核：3*3*128，conv3_1,conv3_2,conv3_3，,256个卷积核，75+2-3+1=75，75*75*256
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        end_points['block3'] = net
        # 最大池化：2*2内取最大值，步长为2，(75+1)/2=38，输出38*38*256
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        # 卷积核：3*3*256，conv4_1,conv4_2,conv4_3，512个卷积核，38+2-3+1=38，输出38*38*512
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        end_points['block4'] = net
        # 最大池化：2*2内取最大值，步长为2，38/2=19，输出19*19*512
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        # 卷积核：3*3*512，conv5_1,conv5_2，conv5_3，512个卷积核，19+2-3+1=19，输出19*19*512
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        end_points['block5'] = net
        # 最大池化：3*3内取最大值，步长为1，19+2-3+1=19，输出19*19*512
        net = slim.max_pool2d(net, [3, 3], stride=1, scope='pool5')

        # 额外卷积层
        # 卷积核：3*3*512（空洞卷积），1024个卷积核，输出19*19*1024
        net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')
        end_points['block6'] = net
        # 对于每个神经元以0.5的概率暂时丢弃，未被丢弃的则/0.5，使得网络相当于一个网络集合的组合，防止过拟合
        net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training)
        # 卷积核：1*1*1024,1024个卷积核，输出19*19*1024
        net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
        end_points['block7'] = net
        # 对于每个神经元以0.5的概率暂时丢弃，使得网络相当于一个网络集合的组合，防止过拟合
        net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training)

        end_point = 'block8'
        with tf.variable_scope(end_point):
            # 卷积核：19*19*1024，256个卷积核，输出1*1*256
            net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
            # padding 长宽各补一个0，输出21*21*1024
            net = common_methods.pad2d(net, pad=(1, 1))
            # 卷积核：3*3*1024,512个卷积核，步长为2，本次不补0（上面补过），输出10*10*512
            net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        end_points[end_point] = net

        end_point = 'block9'
        with tf.variable_scope(end_point):
            # 卷积核：1*1*512,128个卷积核，输出10*10*128
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            # padding 长宽各补一个0，输出12*12*128
            net = common_methods.pad2d(net, pad=(1, 1))
            # 卷积核：3*3*128,256个卷积核，步长为2，本次不补0（上面补过），输出5*5*256
            net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        end_points[end_point] = net
        if '300' in scope:
            end_point = 'block10'
            with tf.variable_scope(end_point):
                # 卷积核：1*1*256,128个卷积核，输出5*5*128
                net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                # 卷积核：3*3*128,256个卷积核，本次不补0，输出3*3*256
                net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
            end_points[end_point] = net

            end_point = 'block11'
            with tf.variable_scope(end_point):
                # 卷积核：1*1*256,128个卷积核，输出3*3*128
                net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                # 卷积核：3*3*128,256个卷积核，本次不补0，输出1*1*256
                net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
            end_points[end_point] = net
        else:
            end_point = 'block10'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                net = common_methods.pad2d(net, pad=(1, 1))
                net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
            end_points[end_point] = net
            end_point = 'block11'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                net = common_methods.pad2d(net, pad=(1, 1))
                net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
            end_points[end_point] = net
            end_point = 'block12'
            with tf.variable_scope(end_point):
                net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                net = common_methods.pad2d(net, pad=(1, 1))
                net = slim.conv2d(net, 256, [4, 4], scope='conv4x4', padding='VALID')
            end_points[end_point] = net

        # 卷积预测层
        logits = []
        predictions = []
        localisations = []
        # i：index  layer：item
        for i, layer in enumerate(feat_layers):
            # 创建layer+'_box'命名空间
            with tf.variable_scope(layer + '_box'):
                # 卷积预测层 p:class l:location
                # 对每个feat_layer，都使用两个3*3的卷积核去预测5个输出（class+location）
                p, l = ssd_network_calc.ssdDefaultboxResult(end_points[layer], num_classes, default_sizes[i],
                                                            default_ratios[i],
                                                            normalizations[i])
            # softmax：将p中元素归一化成和为1的形式，这样可以用来表示概率，处理后添加到列表predictions
            predictions.append(prediction_fn(p))
            # class
            logits.append(p)
            # location
            localisations.append(l)
        return predictions, localisations, logits, end_points


def ssdDefaultboxOne(img_shape, feat_shape, sizes, ratios, step, offset=0.5):
    # y: 0到feat_shape[0]向右复制feat_shape[1]列
    # ·x: 0:到feat_shape[1]向下复制feat_shape[0]行
    # x,y的shape 与feat_shape相同
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(np.float32) + offset) * step / img_shape[0]
    x = (x.astype(np.float32) + offset) * step / img_shape[1]

    # x，y增加第三维.
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)
    # 除去不同的纵横比的defaultbox以外，附加两个aspect ratio为1的
    num_default = len(sizes) + len(ratios)
    h = np.zeros((num_default,), dtype=np.float32)
    w = np.zeros((num_default,), dtype=np.float32)
    # 第一个default box ratio为1   size[0]对应论文的sk
    h[0] = sizes[0] / img_shape[0]
    w[0] = sizes[0] / img_shape[1]
    di = 1
    # sqrt(sk*sk+1)
    if len(sizes) > 1:
        h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
        w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]
        di += 1
    # h=sk/sqrt(ratio)  w=sk*sqrt(ratio)
    for i, r in enumerate(ratios):
        h[i + di] = sizes[0] / img_shape[0] / math.sqrt(r)
        w[i + di] = sizes[0] / img_shape[1] * math.sqrt(r)
    return y, x, h, w


def ssdDefaultboxAll(img_shape, layers_shape, default_sizes, default_ratios, default_steps, offset=0.5):
    layers_default = []
    for i, s in enumerate(layers_shape):
        default_bboxes = ssdDefaultboxOne(img_shape, s, default_sizes[i], default_ratios[i], default_steps[i],
                                          offset=offset)
        layers_default.append(default_bboxes)
    return layers_default
