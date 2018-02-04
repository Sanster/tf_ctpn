import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from layer_utils.generate_anchors import generate_anchors_pre_ctpn
from layer_utils.proposal_layer import proposal_layer
from model.config import cfg


# noinspection PyAttributeOutsideInit,PyProtectedMember
class ctpn:
  def __init__(self, conv):
    self.conv = conv
    self._scope = "CTPN/" + conv._scpoe
    self._predictions = {}
    # self._losses = {}
    # self._anchor_targets = {}
    # self._proposal_targets = {}
    # self._layers = {}
    # self._gt_image = None
    self._act_summaries = []
    # self._score_summaries = {}
    # self._train_summaries = []
    # self._event_summaries = {}
    # self._variables_to_fix = {}

  def setup(self):
    pass

  def create_architecture(self, mode, num_classes, tag=None,
                          anchor_width=16, anchor_h_ratio_step=0.7, num_anchors=10):
    self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    self._im_info = tf.placeholder(tf.float32, shape=[3])
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
    with slim.arg_scope([slim.conv2d, slim.conv2d_in_plane,
                         slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                        weights_regularizer=weights_regularizer,
                        biases_regularizer=biases_regularizer,
                        biases_initializer=tf.constant_initializer(0.0)):
      rois, cls_prob, bbox_pred = self._build_network(training)

    layers_to_output = {'rois': rois}

    for var in tf.trainable_variables():
      self._train_summaries.append(var)

    if testing:
      stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (self._num_classes))
      means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (self._num_classes))
      self._predictions["bbox_pred"] *= stds
      self._predictions["bbox_pred"] += means
    else:
      self._add_losses()
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

  def _build_network(self, is_training=True):
    # select initializers
    if cfg.TRAIN.TRUNCATED:
      initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
    else:
      initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

    net_conv = self.conv._image_to_head(is_training)
    with tf.variable_scope(self._scope, self._scope):
      # build the anchors for the image
      self._anchor_component()
      # region proposal network
      rois = self._region_proposal(net_conv, is_training, initializer)
      # region of interest pooling
      if cfg.POOLING_MODE == 'crop':
        pool5 = self._crop_pool_layer(net_conv, rois, "pool5")
      else:
        raise NotImplementedError

    fc7 = self._head_to_tail(pool5, is_training)
    with tf.variable_scope(self._scope, self._scope):
      # region classification
      cls_prob, bbox_pred = self._region_classification(fc7, is_training,
                                                        initializer, initializer_bbox)

    self._score_summaries.update(self._predictions)

    return rois, cls_prob, bbox_pred

  def _region_proposal(self, net_conv, is_training, initializer):
    rpn = slim.conv2d(net_conv, cfg.RPN_CHANNELS, [3, 3],
                      trainable=is_training, weights_initializer=initializer,
                      scope="rpn_conv/3x3")
    self._act_summaries.append(rpn)

    rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training,
                                weights_initializer=initializer,
                                padding='VALID', activation_fn=None, scope='rpn_cls_score')

    # change it so that the score has 2 as its channel size
    rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
    rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
    rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]), axis=1, name="rpn_cls_pred")
    rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")

    rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training,
                                weights_initializer=initializer,
                                padding='VALID', activation_fn=None, scope='rpn_bbox_pred')

    if is_training:
      rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
      rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")
      # Try to have a deterministic order for the computing graph, for reproducibility
      with tf.control_dependencies([rpn_labels]):
        rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")
    else:
      if cfg.TEST.MODE == 'nms':
        rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
      elif cfg.TEST.MODE == 'top':
        rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
      else:
        raise NotImplementedError

    self._predictions["rpn_cls_score"] = rpn_cls_score
    self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
    self._predictions["rpn_cls_prob"] = rpn_cls_prob
    self._predictions["rpn_cls_pred"] = rpn_cls_pred
    self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
    self._predictions["rois"] = rois

    return rois

  def _anchor_component(self):
    with tf.variable_scope('ANCHOR_' + self._tag) as scope:
      # just to get the shape right
      height = tf.to_int32(tf.ceil(self._im_info[0] / np.float32(self.conv._feat_stride[0])))
      width = tf.to_int32(tf.ceil(self._im_info[1] / np.float32(self.conv._feat_stride[0])))
      anchors, anchor_length = tf.py_func(generate_anchors_pre_ctpn,
                                          [height, width,
                                           self.conv._feat_stride,
                                           cfg.CTPN.ANCHOR_WIDTH, cfg.CTPN.H_RADIO_STEP],
                                          [tf.float32, tf.int32], name="generate_anchors")
      anchors.set_shape([None, 4])
      anchor_length.set_shape([])
      self._anchors = anchors
      self._anchor_length = anchor_length

  def _reshape_layer(self, bottom, num_dim, name):
    input_shape = tf.shape(bottom)
    with tf.variable_scope(name) as scope:
      # change the channel to the caffe format
      to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
      # then force it to have channel 2
      reshaped = tf.reshape(to_caffe,
                            tf.concat(axis=0, values=[[1, num_dim, -1], [input_shape[2]]]))
      # then swap the channel back
      to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
      return to_tf

  def _softmax_layer(self, bottom, name):
    if name.startswith('rpn_cls_prob_reshape'):
      input_shape = tf.shape(bottom)
      bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
      reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
      return tf.reshape(reshaped_score, input_shape)
    return tf.nn.softmax(bottom, name=name)

  def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
    with tf.variable_scope(name) as scope:
      rois, rpn_scores = tf.py_func(proposal_layer,
                                    [rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode,
                                     self._feat_stride, self._anchors, self._num_anchors],
                                    [tf.float32, tf.float32], name="proposal")
      rois.set_shape([None, 5])
      rpn_scores.set_shape([None, 1])

    return rois, rpn_scores
