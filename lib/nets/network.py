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

from layer_utils.generate_anchors import generate_anchors_pre
from layer_utils.proposal_layer import proposal_layer
from layer_utils.proposal_top_layer import proposal_top_layer
from layer_utils.anchor_target_layer import anchor_target_layer
from utils.visualization import draw_bounding_boxes

from model.config import cfg
from abc import abstractmethod


# noinspection PyAttributeOutsideInit,PyProtectedMember,PyMethodMayBeStatic,PyUnresolvedReferences
class Network(object):
    def __init__(self):
        self._predictions = {}
        self._losses = {}
        self._anchor_targets = {}
        self._layers = {}
        self._gt_image = None
        self._act_summaries = []
        self._score_summaries = {}
        self._train_summaries = []
        self._event_summaries = {}
        self._variables_to_fix = {}

    def _add_gt_image(self):
        # add back mean
        image = self._image + cfg.PIXEL_MEANS
        # BGR to RGB (opencv uses BGR)
        resized = tf.image.resize_bilinear(image, tf.to_int32(self._im_info[:2] / self._im_info[2]))
        self._gt_image = tf.reverse(resized, axis=[-1])

    def _add_gt_image_summary(self):
        # use a customized visualization function to visualize the boxes
        if self._gt_image is None:
            self._add_gt_image()
        image = tf.py_func(draw_bounding_boxes,
                           [self._gt_image, self._gt_boxes, self._im_info],
                           tf.float32, name="gt_boxes")

        return tf.summary.image('GROUND_TRUTH', image)

    def _add_act_summary(self, tensor):
        tf.summary.histogram('ACT/' + tensor.op.name + '/activations', tensor)
        tf.summary.scalar('ACT/' + tensor.op.name + '/zero_fraction',
                          tf.nn.zero_fraction(tensor))

    def _add_score_summary(self, key, tensor):
        tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

    def _add_train_summary(self, var):
        tf.summary.histogram('TRAIN/' + var.op.name, var)

    def _reshape_layer(self, input, num_dim, name):
        """
        (1, H, W, Axd) -> (1, H, WxA, d)
        """
        input_shape = tf.shape(input)
        return tf.reshape(input, [input_shape[0], input_shape[1], -1, num_dim], name=name)

    def _softmax_layer(self, input, name):
        # if name.startswith('rpn_cls_prob_reshape'):
        #     input_shape = tf.shape(bottom)
        #     bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
        #     reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
        #     return tf.reshape(reshaped_score, input_shape)
        # return tf.nn.softmax(bottom, name=name)

        input_shape = tf.shape(input)
        return tf.reshape(tf.nn.softmax(tf.reshape(input, [-1, input_shape[3]])),
                          [-1, input_shape[1], input_shape[2], input_shape[3]], name=name)

    def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
        with tf.variable_scope(name) as scope:
            rois, rpn_scores = tf.py_func(proposal_top_layer,
                                          [rpn_cls_prob, rpn_bbox_pred, self._im_info,
                                           self._feat_stride, self._anchors, self._num_anchors],
                                          [tf.float32, tf.float32], name="proposal_top")
            rois.set_shape([cfg.TEST.RPN_TOP_N, 5])
            rpn_scores.set_shape([cfg.TEST.RPN_TOP_N, 1])

        return rois, rpn_scores

    def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
        """
        Do nms -> topN -> apply rpn_bbox_pred to anchors(bbox_transform_inv)
        """
        with tf.variable_scope(name) as scope:
            rois, rpn_scores = tf.py_func(proposal_layer,
                                          [rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode,
                                           self._feat_stride, self._anchors, self._num_anchors],
                                          [tf.float32, tf.float32], name="proposal")
            rois.set_shape([None, 5])
            rpn_scores.set_shape([None, 1])

        return rois, rpn_scores

    def _dropout_layer(self, bottom, name, ratio=0.5):
        return tf.nn.dropout(bottom, ratio, name=name)

    def _anchor_target_layer(self, rpn_cls_score, name):
        """
        Filter percomputed anchors and get anchor labels(positive negative or don't care) by IoU (each anchor's overlap with gt box)
        Only use anchors inside the image
        :param rpn_cls_score: positive and negative score for each anchor
        :return rpn_labels: positive negative or don't care
        """
        with tf.variable_scope(name) as scope:
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
                anchor_target_layer,
                [rpn_cls_score, self._gt_boxes, self._im_info, self._feat_stride, self._anchors, self._num_anchors],
                [tf.float32, tf.float32, tf.float32, tf.float32],
                name="anchor_target")

            rpn_labels.set_shape([1, None, None, self._num_anchors])
            rpn_bbox_targets.set_shape([1, None, None, self._num_anchors * 4])
            rpn_bbox_inside_weights.set_shape([1, None, None, self._num_anchors * 4])
            rpn_bbox_outside_weights.set_shape([1, None, None, self._num_anchors * 4])

            rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
            self._anchor_targets['rpn_labels'] = rpn_labels
            self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
            self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
            self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

            self._score_summaries.update(self._anchor_targets)

        return rpn_labels

    def _anchor_component(self):
        with tf.variable_scope('ANCHOR_' + self._tag) as scope:
            # just to get the shape right
            height = tf.to_int32(tf.ceil(self._im_info[0] / np.float32(self._feat_stride[0])))
            width = tf.to_int32(tf.ceil(self._im_info[1] / np.float32(self._feat_stride[0])))
            anchors, anchor_length = tf.py_func(generate_anchors_pre,
                                                [height, width,
                                                 self._feat_stride,
                                                 cfg.CTPN.ANCHOR_WIDTH, cfg.CTPN.H_RADIO_STEP],
                                                [tf.float32, tf.int32], name="generate_anchors")
            anchors.set_shape([None, 4])
            anchor_length.set_shape([])
            self._anchors = anchors
            self._anchor_length = anchor_length

    def _build_network(self, is_training=True):
        # select initializers
        if cfg.TRAIN.TRUNCATED:
            initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        else:
            initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)

        net_conv = self._image_to_head(is_training)
        with tf.variable_scope(self._scope, self._scope):
            # build the anchors for the image
            self._anchor_component()
            # region proposal network
            self._region_proposal(net_conv, is_training, initializer)

        self._score_summaries.update(self._predictions)

    def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                      + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box

        loss_box = tf.reduce_mean(tf.reduce_sum(
            out_loss_box,
            axis=dim
        ))

        return loss_box

    def smooth_l1_dist(self, deltas, sigma2=9.0, name='smooth_l1_dist'):
        with tf.name_scope(name=name):
            deltas_abs = tf.abs(deltas)
            smoothL1_sign = tf.cast(tf.less(deltas_abs, 1.0 / sigma2), tf.float32)
            return tf.square(deltas) * 0.5 * sigma2 * smoothL1_sign + \
                   (deltas_abs - 0.5 / sigma2) * tf.abs(smoothL1_sign - 1)

    def _build_losses(self, sigma_rpn=3.0):
        with tf.variable_scope('LOSS_' + self._tag) as scope:
            # RPN, class loss
            # (N, H, W x num_anchors, 2) -> (N x H x W x num_anchors, 2)
            rpn_cls_score = tf.reshape(self._predictions['rpn_cls_score_reshape'], [-1, 2])
            # (N, H, W, num_anchors) -> (N x H x W x num_anchors)
            rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1])

            # except don't care label
            rpn_select = tf.where(tf.not_equal(rpn_label, -1))
            # only get positive and negative score/label

            rpn_cls_score = tf.gather(rpn_cls_score, rpn_select)
            rpn_label = tf.gather(rpn_label, rpn_select)

            rpn_cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=rpn_label, logits=rpn_cls_score))

            # RPN, bbox loss
            rpn_bbox_pred = self._predictions['rpn_bbox_pred']  # shape (1, H, W, Ax4)
            rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
            rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
            rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']

            rpn_bbox_pred = tf.gather(tf.reshape(rpn_bbox_pred, [-1, 4]), rpn_select)
            rpn_bbox_targets = tf.gather(tf.reshape(rpn_bbox_targets, [-1, 4]), rpn_select)
            rpn_bbox_inside_weights = tf.gather(tf.reshape(rpn_bbox_inside_weights, [-1, 4]), rpn_select)
            rpn_bbox_outside_weights = tf.gather(tf.reshape(rpn_bbox_outside_weights, [-1, 4]), rpn_select)

            # rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
            #                                     rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1])

            smooth_l1_dist = self.smooth_l1_dist(rpn_bbox_inside_weights * (rpn_bbox_pred - rpn_bbox_targets))
            rpn_loss_box_n = tf.reduce_sum(rpn_bbox_outside_weights * smooth_l1_dist, axis=[1])

            fg_keep = tf.cast(tf.equal(rpn_label, 1), tf.float32)
            rpn_loss_box = tf.reduce_sum(rpn_loss_box_n) / (tf.reduce_sum(tf.cast(fg_keep, tf.float32)) + 1)

            rpn_loss = rpn_cross_entropy + rpn_loss_box

            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            total_loss = tf.add_n(regularization_losses) + rpn_loss

            self._losses['rpn_cross_entropy'] = rpn_cross_entropy
            self._losses['rpn_loss_box'] = rpn_loss_box
            self._losses['rpn_loss'] = rpn_loss
            self._losses['total_loss'] = total_loss

            self._event_summaries.update(self._losses)

        return total_loss

    def _BiLstm(self, input, d_i, d_o, hidden_num, name, initializer, is_training=True):
        img = input
        with tf.variable_scope(name) as scope:
            shape = tf.shape(img)
            N, H, W, C = shape[0], shape[1], shape[2], shape[3]
            img = tf.reshape(img, [N * H, W, C])
            img.set_shape([None, None, d_i])

            lstm_fw_cell = tf.contrib.rnn.LSTMCell(hidden_num, state_is_tuple=True)
            lstm_bw_cell = tf.contrib.rnn.LSTMCell(hidden_num, state_is_tuple=True)

            lstm_out, last_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, img, dtype=tf.float32)
            lstm_out = tf.concat(lstm_out, axis=-1)

            lstm_out = tf.reshape(lstm_out, [N * H * W, 2 * hidden_num])

            print("lstm_out shape")
            print(lstm_out.shape)

            # outputs = slim.fully_connected(lstm_out, d_o,
            #                                weights_initializer=initializer,
            #                                weights_regularizer=tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY),
            #                                trainable=is_training,
            #                                activation_fn=None)

            init_weights = tf.truncated_normal_initializer(stddev=0.1)
            init_biases = tf.constant_initializer(0.0)
            weights = self._make_var('weights', [2 * hidden_num, d_o], init_weights, is_training,
                                     regularizer=self._l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            biases = self._make_var('biases', [d_o], init_biases, is_training)
            outputs = tf.matmul(lstm_out, weights) + biases

            outputs = tf.reshape(outputs, [N, H, W, d_o])

            print("bilstm outputs")
            print(outputs.shape)

            return outputs

    def _l2_regularizer(self, weight_decay=0.0005, scope=None):
        def regularizer(tensor):
            with tf.name_scope(scope, default_name='l2_regularizer', values=[tensor]):
                l2_weight = tf.convert_to_tensor(weight_decay,
                                                 dtype=tensor.dtype.base_dtype,
                                                 name='weight_decay')
                return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')

        return regularizer

    def _make_var(self, name, shape, initializer=None, trainable=True, regularizer=None):
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable, regularizer=regularizer)

    def _lstm_fc(self, input, d_i, d_o, name, trainable=True):
        with tf.variable_scope(name) as scope:
            shape = tf.shape(input)
            N, H, W, C = shape[0], shape[1], shape[2], shape[3]
            input = tf.reshape(input, [N * H * W, C])

            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_biases = tf.constant_initializer(0.0)
            kernel = self._make_var('weights', [d_i, d_o], init_weights, trainable,
                                    regularizer=self._l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            biases = self._make_var('biases', [d_o], init_biases, trainable)

            _O = tf.matmul(input, kernel) + biases
            return tf.reshape(_O, [N, H, W, int(d_o)])

    def _region_proposal(self, net_conv, is_training, initializer):
        rpn = slim.conv2d(net_conv, cfg.RPN_CHANNELS, [3, 3], trainable=is_training, weights_initializer=initializer,
                          scope="rpn_conv/3x3")
        self._act_summaries.append(rpn)

        lstm_output = 512
        bi_lstm = self._BiLstm(rpn, cfg.RPN_CHANNELS, lstm_output, hidden_num=128, name="bi_lstm",
                               initializer=initializer, is_training=is_training)

        # === start use CONV as Fully connect ===
        # # reshape to [N, H, W, C] for 1x1 conv operate
        # shape = tf.shape(rpn)
        # N, H, W, _ = shape[0], shape[1], shape[2], shape[3]
        # bi_lstm_reshape = tf.reshape(bi_lstm, [N, H, W, lstm_output])
        #
        # # use 1x1 conv as FC (N, H, W, num_anchors * 2)
        # rpn_cls_score = slim.conv2d(bi_lstm_reshape, self._num_anchors * 2, [1, 1], trainable=is_training,
        #                             weights_initializer=initializer,
        #                             padding='VALID', activation_fn=None, scope='rpn_cls_score')
        #
        # # (N, H, W, num_anchors * 2) -> (N, H, W * num_anchors, 2)
        # rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
        # rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
        #
        # # get positive text score
        # rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]), axis=1, name="rpn_cls_pred")
        #
        # # (N, H, W*num_anchors, 2) -> (N, H, W, num_anchors*2)
        # rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
        #
        # rpn_bbox_pred = slim.conv2d(bi_lstm_reshape, self._num_anchors * 4, [1, 1], trainable=is_training,
        #                             weights_initializer=initializer,
        #                             padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
        # === end use CONV as Fully connect ===

        # (self.feed('lstm_o').lstm_fc(512,len(anchor_scales) * 10 * 4, name='rpn_bbox_pred'))
        # (self.feed('lstm_o').lstm_fc(512,len(anchor_scales) * 10 * 2,name='rpn_cls_score'))

        rpn_bbox_pred = self._lstm_fc(bi_lstm, 512, self._num_anchors * 4, name='rpn_bbox_pred')

        # (N, H, W, Ax2)
        rpn_cls_score = self._lstm_fc(bi_lstm, 512, self._num_anchors * 2, name='rpn_cls_score')

        # (N, H, WxA, 2)
        rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, name='rpn_cls_score_reshape')

        # (N, H, WxA, 2)
        rpn_cls_prob = self._softmax_layer(rpn_cls_score_reshape, name='rpn_cls_prob')

        # (N, H, W, Ax2)
        rpn_cls_prob_reshape = self._reshape_layer(rpn_cls_prob, self._num_anchors * 2, name='rpn_cls_prob_reshape')

        if is_training:
            self._anchor_target_layer(rpn_cls_score, "anchor")
        else:
            if cfg.TEST.MODE == 'nms':
                rois, _ = self._proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, "rois")
            elif cfg.TEST.MODE == 'top':
                rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            else:
                raise NotImplementedError
            self._predictions["rois"] = rois

        self._predictions["rpn_cls_score"] = rpn_cls_score
        self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
        self._predictions["rpn_cls_prob"] = rpn_cls_prob_reshape
        # self._predictions["rpn_cls_pred"] = rpn_cls_pred
        self._predictions["rpn_bbox_pred"] = rpn_bbox_pred

    @abstractmethod
    def _image_to_head(self, is_training, reuse=None):
        """
        Layers from image input to last Conv layer
        """
        raise NotImplementedError

    def create_architecture(self, mode, num_classes, tag=None,
                            anchor_width=16, anchor_h_ratio_step=0.7, num_anchors=10):
        self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='input')
        self._im_info = tf.placeholder(tf.float32, shape=[3], name='im_info')
        self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self._tag = tag

        self._num_classes = num_classes
        self._mode = mode

        self._anchor_width = anchor_width
        self._anchor_h_ratio_step = anchor_h_ratio_step

        self._num_anchors = num_anchors

        training = mode == 'TRAIN'
        testing = mode == 'TEST'

        assert tag != None

        # handle most of the regularizers here
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
        if cfg.TRAIN.BIAS_DECAY:
            biases_regularizer = weights_regularizer
        else:
            biases_regularizer = tf.no_regularizer

        # list as many types of layers as possible, even if they are not used now
        with arg_scope([slim.conv2d, slim.conv2d_in_plane,
                        slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                       weights_regularizer=weights_regularizer,
                       biases_regularizer=biases_regularizer,
                       biases_initializer=tf.constant_initializer(0.0)):
            self._build_network(training)

        for var in tf.trainable_variables():
            self._train_summaries.append(var)

        layers_to_output = {}
        if testing:
            stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), self._num_anchors)
            means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), self._num_anchors)
            # self._predictions["rpn_bbox_pred"] *= stds
            # self._predictions["rpn_bbox_pred"] += means
        else:
            self._build_losses()
            layers_to_output.update(self._losses)

            val_summaries = []
            with tf.device("/cpu:0"):
                val_summaries.append(self._add_gt_image_summary())
                for key, var in self._event_summaries.items():
                    val_summaries.append(tf.summary.scalar(key, var))
                for key, var in self._score_summaries.items():
                    self._add_score_summary(key, var)
                for var in self._act_summaries:
                    self._add_act_summary(var)
                for var in self._train_summaries:
                    self._add_train_summary(var)

            self._summary_op = tf.summary.merge_all()
            self._summary_op_val = tf.summary.merge(val_summaries)

        layers_to_output.update(self._predictions)

        return layers_to_output

    def get_variables_to_restore(self, variables, var_keep_dic):
        raise NotImplementedError

    def fix_variables(self, sess, pretrained_model):
        raise NotImplementedError

    # Extract the head feature maps, for example for vgg16 it is conv5_3
    # only useful during testing mode
    def extract_head(self, sess, image):
        feed_dict = {self._image: image}
        feat = sess.run(self._layers["head"], feed_dict=feed_dict)
        return feat

    # only useful during testing mode
    def test_image(self, sess, image, im_info):
        feed_dict = {self._image: image,
                     self._im_info: im_info}

        # rois 是 rpn 的输出结果
        rois = sess.run(self._predictions['rois'], feed_dict=feed_dict)
        return rois

    def get_summary(self, sess, blobs):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes']}
        summary = sess.run(self._summary_op_val, feed_dict=feed_dict)

        return summary

    def train_step(self, sess, blobs, train_op):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes']}
        rpn_loss_cls, rpn_loss_box, rpn_loss, total_loss, rpn_bbox_targets, _ = sess.run(
            [self._losses["rpn_cross_entropy"],
             self._losses['rpn_loss_box'],
             self._losses['rpn_loss'],
             self._losses['total_loss'],
             self._anchor_targets['rpn_bbox_targets'],
             train_op],
            feed_dict=feed_dict)

        return rpn_loss_cls, rpn_loss_box, rpn_loss, total_loss, _

    def train_step_with_summary(self, sess, blobs, train_op):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes']}
        rpn_loss_cls, rpn_loss_box, rpn_loss, loss, summary, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                           self._losses['rpn_loss_box'],
                                                                           self._losses['rpn_loss'],
                                                                           self._losses['total_loss'],
                                                                           self._summary_op,
                                                                           train_op],
                                                                          feed_dict=feed_dict)
        return rpn_loss_cls, rpn_loss_box, rpn_loss, loss, summary
