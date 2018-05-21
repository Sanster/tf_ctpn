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
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
import numpy as np

from nets.network import Network
from model.config import cfg


class vgg16(Network):
    def __init__(self):
        Network.__init__(self)
        self._feat_stride = [16, ]
        self._feat_compress = [1. / float(self._feat_stride[0]), ]
        self._scope = 'vgg_16'

    def _image_to_head(self, is_training, reuse=None):
        with tf.variable_scope(self._scope, self._scope, reuse=reuse):
            net = slim.repeat(self._image, 2, slim.conv2d, 64, [3, 3],
                              trainable=True, scope='conv1')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],
                              trainable=True, scope='conv2')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3],
                              trainable=is_training, scope='conv3')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                              trainable=is_training, scope='conv4')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                              trainable=is_training, scope='conv5')

        self._act_summaries.append(net)
        self._layers['head'] = net

        return net

    def get_variables_to_restore(self, variables, var_keep_dic):
        variables_to_restore = []

        for v in variables:
            # exclude the conv weights that are fc weights in vgg16
            if v.name == (self._scope + '/fc6/weights:0') or \
                    v.name == (self._scope + '/fc7/weights:0'):
                self._variables_to_fix[v.name] = v
                continue
            # exclude the first conv layer to swap RGB to BGR
            if v.name == (self._scope + '/conv1/conv1_1/weights:0'):
                self._variables_to_fix[v.name] = v
                continue
            if v.name.split(':')[0] in var_keep_dic:
                print('Variables restored: %s' % v.name)
                variables_to_restore.append(v)

        return variables_to_restore

    def fix_variables(self, sess, pretrained_model):
        print('Fix VGG16 layers..')
        with tf.variable_scope('Fix_VGG16') as scope:
            with tf.device("/cpu:0"):
                # fix RGB to BGR
                # We only need to reverse the first layer of conv, conv2 is the sum of conv1
                conv1_rgb = tf.get_variable("conv1_rgb", [3, 3, 3, 64], trainable=False)
                restorer_fc = tf.train.Saver({self._scope + "/conv1/conv1_1/weights": conv1_rgb})
                restorer_fc.restore(sess, pretrained_model)
                sess.run(tf.assign(self._variables_to_fix[self._scope + '/conv1/conv1_1/weights:0'],
                                   tf.reverse(conv1_rgb, [2])))
