"""
Visualization anchors on scaled image
"""
import _init_paths
import os
import argparse

import numpy as np
import cv2

from layer_utils.generate_anchors import generate_anchors


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', default='')
    parser.add_argument('--scale', type=int, default=600)
    parser.add_argument('--max_scale', default=1200)
    args = parser.parse_args()

    if not os.path.exists(args.img):
        parser.error('Image not exist.')
    return args


def draw_anchors(img, heights, width, start_center):
    anchors = []
    center = start_center
    for height in heights:
        anchors.append((
            center[0] - width // 2,
            center[1] - height // 2,
            center[0] + width // 2,
            center[1] + height // 2
        ))
        center = (center[0] + width, center[1])

    for anchor in anchors:
        img = cv2.rectangle(img, (anchor[0], anchor[1]), (anchor[2], anchor[3]), color=(255, 0, 0))
    return img


if __name__ == '__main__':
    args = parse_args()
    img = cv2.imread(args.img)
    im_size_min = min(img.shape)
    im_size_max = max(img.shape)

    im_scale = float(args.scale) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > args.max_scale:
        im_scale = float(args.max_scale) / float(im_size_max)

    img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    print("Scaled image size")
    print(img.shape)
    width = img.shape[1]
    height = img.shape[0]

    base_anchors = generate_anchors(base_height=11,
                                    num_anchors=10,
                                    anchor_width=16,
                                    h_ratio_step=0.7)

    heights = [x[3] - x[1] for x in base_anchors]

    img = draw_anchors(img, heights, 16, (width // 2, height // 2))
    img = draw_anchors(img, heights, 16, (100, 150))
    img = draw_anchors(img, heights, 16, (width - 300, 150))
    img = draw_anchors(img, heights, 16, (100, height - 150))
    img = draw_anchors(img, heights, 16, (width - 300, height - 150))

    cv2.namedWindow('test')
    cv2.imshow('test', img)
    cv2.waitKey()
