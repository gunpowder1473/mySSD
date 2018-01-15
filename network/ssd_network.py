import math
from collections import namedtuple
import numpy as np
import tensorflow as tf
from Common import common_methods
from network import ssd_network_struct, ssd_network_calc
from Bboxes import bboxes_method

slim = tf.contrib.slim

# list中的元素可以以SSDParameters的属性的方式访问（只读）
SSDParams = namedtuple('SSDParameters', ['img_shape',
                                         'num_classes',
                                         'no_annotation_label',
                                         'feat_layers',
                                         'feat_shapes',
                                         'default_size_bounds',
                                         'default_sizes',
                                         'default_ratios',
                                         'default_steps',
                                         'default_offset',
                                         'normalizations',
                                         'prior_scaling'
                                         ])


class SSDNet_300(object):
    default_params = SSDParams(
        img_shape=(300, 300),
        num_classes=30,
        no_annotation_label=30,
        feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11'],
        feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
        default_size_bounds=[0.15, 0.90],
        default_sizes=[(21., 45.),  # 0.07*300
                       (45., 99.),  # 0.15*300
                       (99., 153.),  # 0.33*300
                       (153., 207.),  # 0.51*300
                       (207., 261.),  # 0.69*300
                       (261., 315.)],  # 0.87*300
        default_ratios=[[2, .5],
                        [2, .5, 3, 1. / 3],
                        [2, .5, 3, 1. / 3],
                        [2, .5, 3, 1. / 3],
                        [2, .5],
                        [2, .5]],
        # 对应于上面的default_layers的block输出的feature map上每移动一个元素，相当于原图
        # 移动default_steps个元素，这将在default_box的location的offset计算中体现为一个乘法因子，如：8 ≈ 300 / 38
        default_steps=[8, 16, 32, 64, 100, 300],
        default_offset=0.5,
        normalizations=[20, -1, -1, -1, -1, -1],
        prior_scaling=[0.1, 0.1, 0.2, 0.2]
    )

    def __init__(self, params=None):
        if isinstance(params, SSDParams):
            self.params = params
        else:
            self.params = SSDNet_300.default_params

    def argScope(self, weight_decay=0.0005, data_format='NHWC'):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                padding='SAME',
                                data_format=data_format):
                with slim.arg_scope([common_methods.pad2d,
                                     common_methods.l2_normalization,
                                     common_methods.dataFormatChange],
                                    data_format=data_format) as sc:
                    return sc

    def creatNetwork(self, inputs, is_training=True, dropout_keep_prob=0.5, prediction_fn=slim.softmax,
                     reuse=None, scope='ssd_300'):
        network = ssd_network_struct.SSDNet(inputs,
                                                num_classes=self.params.num_classes,
                                                feat_layers=self.params.feat_layers,
                                                default_sizes=self.params.default_sizes,
                                                default_ratios=self.params.default_ratios,
                                                normalizations=self.params.normalizations,
                                                is_training=is_training,
                                                dropout_keep_prob=dropout_keep_prob,
                                                prediction_fn=prediction_fn,
                                                reuse=reuse,
                                                scope=scope)
        shapes = ssdFeatureShape(network[0], self.params.feat_shapes)
        self.params = self.params._replace(feat_shapes=shapes)
        return network

    def creatDefaultbox(self, img_shape):
        return ssd_network_struct.ssdDefaultboxAll(img_shape, self.params.feat_shapes, self.params.default_sizes,
                                                   self.params.default_ratios, self.params.default_steps,
                                                   self.params.default_offset)

    def defaultBboxesEncodeAllLayer(self, labels, bboxes, layer, scope='ssd_bboxes_Encode'):
        return bboxes_method.bboxesEncodeAllLayer(labels, bboxes, layer, self.params.num_classes,
                                                  prior_scaling=self.params.prior_scaling, scope=scope)

    def defaultBboxesDecode(self, feat_localizations, layer, scope='ssd_bboxes_decode'):
        return bboxes_method.bboxDecodeAllLayers(feat_localizations, layer,
                                                 prior_scaling=self.params.prior_scaling, scope=scope)

    def losses(self, logits, localisations, gclasses, glocalisations, gscores, match_threshold=0.5,
               negative_ratio=3., alpha=1., label_smoothing=0., scope='ssd_losses'):
        return ssd_network_calc.ssd_losses(logits, localisations, gclasses, glocalisations, gscores,
                                           match_threshold=match_threshold, negative_ratio=negative_ratio, alpha=alpha,
                                           scope=scope)


class SSDNet_512(object):
    default_params = SSDParams(
        img_shape=(512, 512),
        num_classes=30,
        no_annotation_label=30,
        feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11', 'block12'],
        feat_shapes=[(64, 64), (32, 32), (16, 16), (8, 8), (4, 4), (2, 2), (1, 1)],
        default_size_bounds=[0.1, 0.90],
        default_sizes=[(20.48, 51.2),
                       (51.2, 133.12),
                       (133.12, 215.04),
                       (215.04, 296.96),
                       (296.96, 378.88),
                       (378.88, 460.8),
                       (460.8, 542.72)],
        default_ratios=[[2, .5],
                        [2, .5, 3, 1. / 3],
                        [2, .5, 3, 1. / 3],
                        [2, .5, 3, 1. / 3],
                        [2, .5, 3, 1. / 3],
                        [2, .5],
                        [2, .5]],
        # 对应于上面的default_layers的block输出的feature map上每移动一个元素，相当于原图
        # 移动default_steps个元素，这将在default_box的location的offset计算中体现为一个乘法因子，如：8 ≈ 300 / 38
        default_steps=[8, 16, 32, 64, 128, 256, 512],
        default_offset=0.5,
        normalizations=[20, -1, -1, -1, -1, -1, -1],
        prior_scaling=[0.1, 0.1, 0.2, 0.2]
    )

    def __init__(self, params=None):
        if isinstance(params, SSDParams):
            self.params = params
        else:
            self.params = SSDNet_512.default_params

    def argScope(self, weight_decay=0.0005, data_format='NHWC'):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                padding='SAME',
                                data_format=data_format):
                with slim.arg_scope([common_methods.pad2d,
                                     common_methods.l2_normalization,
                                     common_methods.dataFormatChange],
                                    data_format=data_format) as sc:
                    return sc

    def creatNetwork(self, inputs, is_training=True, dropout_keep_prob=0.5, prediction_fn=slim.softmax,
                     reuse=None, scope='ssd_512'):
        network = ssd_network_struct.SSDNet(inputs,
                                            num_classes=self.params.num_classes,
                                            feat_layers=self.params.feat_layers,
                                            default_sizes=self.params.default_sizes,
                                            default_ratios=self.params.default_ratios,
                                            normalizations=self.params.normalizations,
                                            is_training=is_training,
                                            dropout_keep_prob=dropout_keep_prob,
                                            prediction_fn=prediction_fn,
                                            reuse=reuse,
                                            scope=scope)
        shapes = ssdFeatureShape(network[0], self.params.feat_shapes)
        self.params = self.params._replace(feat_shapes=shapes)
        return network

    def creatDefaultbox(self, img_shape):
        return ssd_network_struct.ssdDefaultboxAll(img_shape, self.params.feat_shapes, self.params.default_sizes,
                                                   self.params.default_ratios, self.params.default_steps,
                                                   self.params.default_offset)

    def defaultBboxesEncodeAllLayer(self, labels, bboxes, layer, scope='ssd_bboxes_Encode'):
        return bboxes_method.bboxesEncodeAllLayer(labels, bboxes, layer, self.params.num_classes,
                                                  prior_scaling=self.params.prior_scaling, scope=scope)

    def defaultBboxesDecode(self, feat_localizations, layer, scope='ssd_bboxes_decode'):
        return bboxes_method.bboxDecodeAllLayers(feat_localizations, layer,
                                                 prior_scaling=self.params.prior_scaling, scope=scope)

    def losses(self, logits, localisations, gclasses, glocalisations, gscores, match_threshold=0.5,
               negative_ratio=3., alpha=1., scope='ssd_losses'):
        return ssd_network_calc.ssd_losses(logits, localisations, gclasses, glocalisations, gscores,
                                           match_threshold=match_threshold, negative_ratio=negative_ratio, alpha=alpha,
                                           scope=scope)


def ssdFeatureShape(predictions, default_shapes=None):
    feat_shapes = []
    for l in predictions:
        if isinstance(l, np.ndarray):
            shape = l.shape
        else:
            shape = l.get_shape().as_list()
        # shape[0]是batchsize
        shape = shape[1:4]
        if None in shape:
            return default_shapes
        else:
            feat_shapes.append(shape)
    return feat_shapes
