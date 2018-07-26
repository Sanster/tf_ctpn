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
from nets.mobilenet import conv_blocks as ops
from nets.mobilenet import mobilenet as lib
from nets.mobilenet import mobilenet_v2 as mobilenet_v2

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

        self.variables_to_restore = slim.get_variables_to_restore()

        self._act_summaries.append(net)
        self._layers['head'] = net

        return net

    def get_variables_to_restore(self, variables, var_keep_dic):
        pass
