#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import time

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms
from text_connector import TextDetector

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1

CLASSES = ('__background__', 'text')


def vis_detections(im, im_name, class_name, dets, result_dir, thresh=0.5, text=False):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )

        if text:
            ax.text(bbox[0], bbox[1] - 2,
                    '{:s} {:.3f}'.format(class_name, score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                 fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.savefig(os.path.join(result_dir, im_name))
    plt.show()


def draw_rpn_boxes(img, img_name, boxes, scores, nms, save_dir):
    """
    :param boxes: [(x1, y1, x2, y2)]
    """
    base_name = img_name.split('/')[-1]
    color = (0, 255, 0)
    out = img.copy()

    if nms:
        boxes, scores = TextDetector.pre_process(boxes, scores)
        file_name = "%s_rpn_nms.jpg" % base_name
    else:
        file_name = "%s_rpn.jpg" % base_name

    for i, box in enumerate(boxes):
        cv2.rectangle(out, (box[0], box[1]), (box[2], box[3]), color, 2)
        cx = int((box[0] + box[2]) / 2)
        cy = int((box[1] + box[3]) / 2)
        cv2.putText(out, "%.01f" % scores[i], (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0))

    cv2.imwrite(os.path.join(save_dir, file_name), out)


def demo(sess, net, im_file, result_dir):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(
        timer.diff, boxes.shape[0]))

    img_name = im_file.split('/')[-1]
    draw_rpn_boxes(im, img_name, boxes, scores[:, np.newaxis], True, result_dir)
    draw_rpn_boxes(im, img_name, boxes, scores[:, np.newaxis], False, result_dir)

    # Run TextDetector to merge small box
    line_detector = TextDetector()
    text_lines = line_detector.detect(boxes, scores[:, np.newaxis], im.shape[:2])
    print("Detect %d text lines" % len(text_lines))

    boxes = np.hstack((text_lines[:, 0:2], text_lines[:, 6:8]))
    scores = text_lines[:, -1:]

    # Visualize detections
    dets = np.hstack((boxes, scores)).astype(np.float32)
    vis_detections(im, img_name, CLASSES[1], dets, result_dir, thresh=0)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow CTPN demo')
    parser.add_argument('--net', dest='net', choices=['vgg16'], default='vgg16')
    parser.add_argument('--img_dir', default='./data/demo')
    parser.add_argument('--dataset', dest='dataset', help='model tag', default='voc_2007_trainval')
    parser.add_argument('--tag', dest='tag', help='model tag', default='default')
    args = parser.parse_args()

    if not os.path.exists(args.img_dir):
        print("img dir not exists.")
        exit(-1)

    args.result_dir = os.path.join('./data/result', args.tag)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    return args


if __name__ == '__main__':
    args = parse_args()

    # model path
    netname = args.net
    dataset = args.dataset

    ckpt_dir = os.path.join('output', netname, dataset, args.tag)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if netname == 'vgg16':
        net = vgg16()
    elif netname == 'res101':
        net = resnetv1(num_layers=101)
    elif netname == 'mobile':
        net = mobilenetv1()
    else:
        raise NotImplementedError

    net.create_architecture("TEST",
                            num_classes=len(CLASSES),
                            tag='default',
                            anchor_width=16,
                            anchor_h_ratio_step=0.7,
                            num_anchors=10)
    saver = tf.train.Saver()
    saver.restore(sess, ckpt.model_checkpoint_path)

    print('Loaded network {:s}'.format(ckpt.model_checkpoint_path))

    im_files = glob.glob(args.img_dir + "/*.*")
    for im_file in im_files:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for {}'.format(im_file))
        demo(sess, net, im_file, args.result_dir)
