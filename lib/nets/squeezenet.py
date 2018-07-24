import tensorflow as tf
import tensorflow.contrib.slim as slim

from nets.network import Network


class SqueezeNet(Network):
    def __init__(self):
        super().__init__()
        self._feat_stride = [16, ]
        self._scope = 'squeezenet'

    def _arg_scope(self, is_training, reuse=None):
        weight_decay = 0.0
        keep_probability = 1.0

        batch_norm_params = {
            'is_training': is_training,
            # Decay for the moving averages.
            'decay': 0.995,
            # epsilon to prevent 0s in variance.
            'epsilon': 0.001
        }

        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_initializer=slim.xavier_initializer_conv2d(uniform=True),
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            with tf.variable_scope(self._scope, self._scope, reuse=reuse):
                with slim.arg_scope([slim.batch_norm, slim.dropout],
                                    is_training=is_training) as sc:
                    return sc

    def get_variables_to_restore(self, variables, var_keep_dic):
        pass

    def _image_to_head(self, is_training, reuse=None):
        with slim.arg_scope(self._arg_scope(is_training, reuse)):
            net = slim.conv2d(self._image, 96, [3, 3], stride=1, scope='conv1')
            net = slim.max_pool2d(net, [2, 2], stride=2, scope='maxpool1')
            net = self.fire_module(net, 16, 64, scope='fire2')
            net = self.fire_module(net, 16, 64, scope='fire3')
            net = self.fire_module(net, 32, 128, scope='fire4')
            net = slim.max_pool2d(net, [2, 2], stride=2, scope='maxpool4')
            net = self.fire_module(net, 32, 128, scope='fire5')
            net = self.fire_module(net, 48, 192, scope='fire6')
            net = self.fire_module(net, 48, 192, scope='fire7')
            net = self.fire_module(net, 64, 256, scope='fire8')
            net = slim.max_pool2d(net, [2, 2], stride=2, scope='maxpool8', padding='SAME')
            net = self.fire_module(net, 64, 256, scope='fire9')
            net = slim.max_pool2d(net, [2, 2], stride=2, scope='maxpool9', padding='SAME')
            net = self.fire_module(net, 64, 512, scope='fire10')

        self._act_summaries.append(net)
        self._layers['head'] = net

        return net

    def fire_module(self, inputs,
                    squeeze_depth,
                    expand_depth,
                    reuse=None,
                    scope=None,
                    outputs_collections=None):
        with tf.variable_scope(scope, 'fire', [inputs], reuse=reuse):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                outputs_collections=None):
                net = self.squeeze(inputs, squeeze_depth)
                outputs = self.expand(net, expand_depth)
                return outputs

    def squeeze(self, inputs, num_outputs):
        return slim.conv2d(inputs, num_outputs, [1, 1], stride=1, scope='squeeze')

    def expand(self, inputs, num_outputs):
        with tf.variable_scope('expand'):
            e1x1 = slim.conv2d(inputs, num_outputs, [1, 1], stride=1, scope='1x1')
            e3x3 = slim.conv2d(inputs, num_outputs, [3, 3], scope='3x3')
        return tf.concat([e1x1, e3x3], 3)
