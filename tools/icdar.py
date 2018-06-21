#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import time
from zipfile import ZipFile

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


def demo(sess, net, im_file, icdar_dir, oriented=False):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()

    # Run TextDetector to merge small box
    line_detector = TextDetector(oriented)

    # text_lines point order: left-top, right-top, left-bottom, right-bottom
    text_lines = line_detector.detect(boxes, scores[:, np.newaxis], im.shape[:2])
    print("Image %s, detect %d text lines in %.3fs" % (im_file, len(text_lines), timer.diff))

    return save_ICDAR15_result(text_lines, icdar_dir, im_file)


def save_ICDAR15_result(text_lines, icdar_dir, im_file):
    # ICDAR15 script.py need clockwise box
    boxes = [[l[0], l[1], l[2], l[3], l[6], l[7], l[4], l[5]] for l in text_lines]

    im_name = im_file.split('/')[-1].split('.')[0]
    res_file = os.path.join(icdar_dir, 'res_%s.txt' % im_name)
    if not os.path.exists(icdar_dir):
        os.makedirs(icdar_dir)

    with open(res_file, mode='w') as f:
        for line in boxes:
            f.write('%d,%d,%d,%d,%d,%d,%d,%d\n' % (line[0], line[1], line[2], line[3],
                                                   line[4], line[5], line[6], line[7]))
    return res_file


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow CTPN demo')
    parser.add_argument('--net', dest='net', choices=['vgg16'], default='vgg16')
    parser.add_argument('--img_dir', default='./data/demo')
    parser.add_argument('--dataset', dest='dataset', help='model tag', default='voc_2007_trainval')
    parser.add_argument('--tag', dest='tag', help='model tag', default='default')
    parser.add_argument('-o', '--oriented', action='store_true', default=False, help='output rotated detect box')
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

    cfg.USE_GPU_NMS = True
    net.create_architecture("TEST",
                            num_classes=len(CLASSES),
                            tag=args.tag,
                            anchor_width=cfg.CTPN.ANCHOR_WIDTH,
                            anchor_h_ratio_step=cfg.CTPN.H_RADIO_STEP,
                            num_anchors=cfg.CTPN.NUM_ANCHORS)
    saver = tf.train.Saver()
    saver.restore(sess, ckpt.model_checkpoint_path)

    print('Loaded network {:s}'.format(ckpt.model_checkpoint_path))

    txt_files = []
    icdar_dir = os.path.join(args.result_dir, 'ICDAR15')

    im_files = glob.glob(args.img_dir + "/*.*")
    for im_file in im_files:
        txt_file = demo(sess, net, im_file, icdar_dir, args.oriented)
        txt_files.append(txt_file)

    zip_path = os.path.join('./tools/ICDAR15', 'submit.zip')
    print(os.path.abspath(zip_path))
    with ZipFile(zip_path, 'w') as f:
        for txt in txt_files:
            f.write(txt, txt.split('/')[-1])

