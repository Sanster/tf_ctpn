# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from lib.nets.mobilenet import conv_blocks as ops
from lib.nets.mobilenet import mobilenet as lib
from lib.nets.mobilenet import mobilenet_v2 as mobilenet_v2

from nets.network import Network
from model.config import cfg


class MobileNetV2(Network):
    def __init__(self):
        Network.__init__(self)
        self._feat_stride = [32, ]
        self._scope = 'mobilenet_v2'

    def _image_to_head(self, is_training, reuse=None):
        with slim.arg_scope(mobilenet_v2.training_scope(is_training=is_training)):
            net, endpoints = mobilenet_v2.mobilenet_base(self._image)

        self._act_summaries.append(net)
        self._layers['head'] = net

        return net

    def get_variables_to_restore(self, variables, var_keep_dic):
        pass

    def reverse_RGB_weights(self, sess, pretrained_model):
        print('Fix MobileNet V2 layers..')
        with tf.variable_scope('Fix_MobileNet_V2') as scope:
            with tf.device("/cpu:0"):
                # fix RGB to BGR, and match the scale by (255.0 / 2.0)
                Conv2d_0_rgb = tf.get_variable("Conv2d_0_rgb", [3, 3, 3, 32], trainable=False)
                restorer_fc = tf.train.Saver({self._scope + "/Conv2d_0/weights": Conv2d_0_rgb})
                restorer_fc.restore(sess, pretrained_model)

                sess.run(tf.assign(self._variables_to_fix[self._scope + "/Conv2d_0/weights:0"],
                                   tf.reverse(Conv2d_0_rgb / (255.0 / 2.0), [2])))
