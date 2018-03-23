#!/usr/env/bin python3

import tensorflow as tf
import argparse
import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
from easydict import EasyDict as edict

from demo import vis_detections
from layer_utils.generate_anchors import generate_anchors_pre_ctpn
from layer_utils.proposal_layer import proposal_layer
from model.bbox_transform import bbox_transform_inv
from model.config import cfg
from model.nms_wrapper import nms
from model.test import _get_blobs, _clip_boxes
from text_connector import TextDetector
from utils.timer import Timer

CLASSES = ('__background__', 'text')


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def)
    return graph


def demo(sess, fetches, feeds, im_file, classes):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(im_file)

    blobs, im_scales = _get_blobs(im)
    im_blob = blobs['data']
    im_info = np.array([im_blob.shape[1], im_blob.shape[2],
                        im_scales[0]], dtype=np.float32)

    timer = Timer()
    timer.tic()
    # run RPN
    RPN_fetches = [fetches.rpn_cls_prob, fetches.rpn_bbox_pred]
    RPN_feed_dict = {feeds.input: im_blob}

    rpn_cls_prob, rpn_bbox_pred = sess.run(
        RPN_fetches, feed_dict=RPN_feed_dict)

    height = rpn_cls_prob.shape[1]
    width = rpn_cls_prob.shape[2]
    stride = [16, ]

    anchors, anchor_lenght = generate_anchors_pre_ctpn(height, width,
                                                       stride,
                                                       anchor_width=16,
                                                       anchor_h_ratio_step=0.7)

    rois, roi_scores = proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info,
                                      cfg_key="TEST",
                                      _feat_stride=[16, ],
                                      anchors=anchors,
                                      num_anchors=10)

    roi_boxes = rois[:, 1:5] / im_scales[0]

    textDectctor = TextDetector()
    textLines = textDectctor.detect(
        roi_boxes, roi_scores, im.shape[:2])
    boxes = np.hstack((textLines[:, 0:2], textLines[:, 6:8]))
    scores = textLines[:, -1:]

    # Visualize detections for each class
    CONF_THRESH = 0.7
    cls = 'text'
    dets = np.hstack((boxes, scores)).astype(np.float32)

    vis_detections(im, cls, dets, thresh=CONF_THRESH)

    print('total time: {:.3f}s'.format(timer.total_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pb_file", default="./output/model/ctpn.pb", type=str,
                        help="Frozen model file to import")
    parser.add_argument("--img_dir", default="./data/demo",
                        type=str, help="Image dir to detect")
    args = parser.parse_args()

    if not os.path.exists(args.pb_file):
        print("pb file not exists.")
        exit(-1)

    if not os.path.exists(args.img_dir):
        print("img dir not exists.")
        exit(-1)

    print("Load pb file: %s" % args.pb_file)

    graph = load_graph(args.pb_file)

    # for v in graph.get_operations():
    #     print(v.name)

    fetches = edict()
    fetches.rpn_cls_prob = graph.get_tensor_by_name(
        'import/MobilenetV1_2/rpn_cls_prob/transpose_1:0')
    fetches.rpn_bbox_pred = graph.get_tensor_by_name(
        'import/MobilenetV1_2/rpn_bbox_pred/BiasAdd:0')
    fetches.cls_prob = graph.get_tensor_by_name(
        'import/MobilenetV1_4/cls_prob:0')
    fetches.bbox_pred = graph.get_tensor_by_name('import/add:0')

    feeds = edict()
    feeds.input = graph.get_tensor_by_name('import/input:0')
    feeds.im_info = graph.get_tensor_by_name('import/im_info:0')
    feeds.rois = graph.get_tensor_by_name(
        'import/MobilenetV1_2/rois/proposal:0')

    im_files = glob.glob(args.img_dir + "/*.*")
    with tf.Session(graph=graph) as sess:
        for im_file in im_files:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('Demo for {}'.format(im_file))
            demo(sess, fetches, feeds, im_file, CLASSES)

