import os
import numpy as np
import math
import cv2 as cv

# path = '/media/D/code/OCR/text-detection-ctpn/data/mlt_english+chinese/image'
# gt_path = '/media/D/code/OCR/text-detection-ctpn/data/mlt_english+chinese/label'

path = '/home/cwq/data/ICDAR13/Challenge2_Training_Task12_Images'
gt_path = '/home/cwq/data/ICDAR13/Challenge2_Training_Task1_GT'

out_path = '/home/cwq/data/ICDAR13/Challenge2_Training_Task12_Images_splited'
label_out_path = '/home/cwq/data/ICDAR13/Challenge2_Training_Task1_GT_splited'
if not os.path.exists(out_path):
    os.makedirs(out_path)
files = os.listdir(path)
files.sort()
# files=files[:100]
for file in files:
    _, basename = os.path.split(file)
    if basename.lower().split('.')[-1] not in ['jpg', 'png']:
        continue
    stem, ext = os.path.splitext(basename)
    gt_file = os.path.join(gt_path, 'gt_' + stem + '.txt')
    img_path = os.path.join(path, file)
    print(img_path)
    img = cv.imread(img_path)
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1200) / float(im_size_max)
    re_im = cv.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv.INTER_AREA)
    re_size = re_im.shape
    cv.imwrite(os.path.join(out_path, stem) + '.jpg', re_im)

    with open(gt_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        splitted_line = line.strip().lower().split(' ')
        splitted_line = [int(n) for n in splitted_line[:-1]]

        xmin = int(splitted_line[0] * im_scale)
        ymin = int(splitted_line[1] * im_scale)
        xmax = int(splitted_line[2] * im_scale)
        ymax = int(splitted_line[3] * im_scale)

        if xmin < 0:
            xmin = 0
        if xmax > re_size[1] - 1:
            xmax = re_size[1] - 1
        if ymin < 0:
            ymin = 0
        if ymax > re_size[0] - 1:
            ymax = re_size[0] - 1

        width = xmax - xmin
        height = ymax - ymin

        # 将完整的文字区域切分为宽度为 16 的小区域
        step = 16.0
        xmins = []

        anchor_count = int(math.ceil(width / step))
        for i in range(anchor_count):
            xmins.append(i * int(step) + xmin)

        if not os.path.exists(label_out_path):
            os.makedirs(label_out_path)

        with open(os.path.join(label_out_path, "gt_" + stem) + '.txt', 'a') as f:
            for i in range(len(xmins)):
                f.writelines(str(xmins[i]))
                f.writelines(" ")
                f.writelines(str(int(ymin)))
                f.writelines(" ")
                # anchor box 的宽度为 16,
                f.writelines(str(int(xmins[i] + step - 1)))
                f.writelines(" ")
                f.writelines(str(int(ymax)))
                f.writelines("\n")

        # reimplement
        # step = 16.0
        # x_left = []
        # x_right = []
        # x_left.append(xmin)
        # x_left_start = int(math.ceil(xmin / 16.0) * 16.0)
        # if x_left_start == xmin:
        #     x_left_start = xmin + 16
        # for i in np.arange(x_left_start, xmax, 16):
        #     x_left.append(i)
        # x_left = np.array(x_left)
        #
        # x_right.append(x_left_start - 1)
        # for i in range(1, len(x_left) - 1):
        #     x_right.append(x_left[i] + 15)
        # x_right.append(xmax)
        # x_right = np.array(x_right)
        #
        # idx = np.where(x_left == x_right)
        # x_left = np.delete(x_left, idx, axis=0)
        # x_right = np.delete(x_right, idx, axis=0)
        #
        # if not os.path.exists(label_out_path):
        #     os.makedirs(label_out_path)
        #
        # with open(os.path.join(label_out_path, "gt_" + stem) + '.txt', 'a') as f:
        #     for i in range(len(x_left)):
        #         f.writelines(str(int(x_left[i])))
        #         f.writelines(" ")
        #         f.writelines(str(int(ymin)))
        #         f.writelines(" ")
        #         f.writelines(str(int(x_right[i])))
        #         f.writelines(" ")
        #         f.writelines(str(int(ymax)))
        #         f.writelines("\n")
