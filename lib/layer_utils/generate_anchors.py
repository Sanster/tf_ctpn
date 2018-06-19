# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def generate_anchors(base_height=11, num_anchors=10, anchor_width=16, h_ratio_step=0.7):
    """
    Generate anchor windows template by using different hight start from base_size
    According to the ctpn paper, anchor's width is always 16 pixels

    Anchor heights in ctpn sorce code: [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
    """
    base_anchor = np.array([1, 1, anchor_width, anchor_width]) - 1
    h_ratios = h_ratio_step ** np.arange(0, num_anchors)

    w, h, x_ctr, y_ctr = _whctrs(base_anchor)
    ws = np.array([16 for _ in range(num_anchors)])

    hs = np.ceil(base_height / h_ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def generate_anchors_pre(height, width, feat_stride, num_anchors=10, anchor_width=16, anchor_h_ratio_step=0.7):
    """
    A wrapper function to generate anchors given by different height scale
    :arg
      height/width: height/width of last shared cnn layer feature map
      feat_stride: total stride until the last shared cnn layer

    :returns
      anchors: anchors on input image
      length: The total number of anchors
    """
    # print("width: %d, height: %d" %(width,height))
    anchors = generate_anchors(num_anchors=num_anchors, h_ratio_step=anchor_h_ratio_step, anchor_width=anchor_width)
    A = anchors.shape[0]
    shift_x = np.arange(0, width) * feat_stride
    shift_y = np.arange(0, height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
    K = shifts.shape[0]
    # width changes faster, so here it is H, W, C
    anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False)
    length = np.int32(anchors.shape[0])

    return anchors, length


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - ws / 2,
                         y_ctr - hs / 2,
                         x_ctr + ws / 2,
                         y_ctr + hs / 2)).astype(np.int32)
    return anchors


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


if __name__ == '__main__':
    import time

    t = time.time()
    anchors = generate_anchors(base_height=11, num_anchors=6, anchor_width=16, h_ratio_step=0.7)
    print(anchors)
    for anchor in anchors:
        print(anchor[3] - anchor[1])

    # c, length = generate_anchors_pre(47, 37, 16, 6)
    # print(c)
    # print(length)
