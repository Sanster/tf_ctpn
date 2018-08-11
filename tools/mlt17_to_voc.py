import os
import numpy as np
import math
import cv2
from tools.convert_utils import build_voc_dirs, generate_xml

"""
gt 样例
矩形四点坐标的顺序： left-top, right-top, right-bottom, left-bottom
x1,y1,x2,y2,x3,y3,x4,y4,language,text

language 可能有：
- Arabic
- Latin
- Chinese
- Korean
- Japanese
- Bangla
- Symbols
- None

text 可能有：
- 图片上对应的文字
- 无法识别的文字用 ### 表示

只转换 Latin 和 Chinese 语言，忽略 ### 的部分
"""

DEBUG = False

img_dir = '/home/cwq/data/MLT2017/val'
gt_dir = '/home/cwq/data/MLT2017/val_gt'

out_dir = '/home/cwq/data/MLT2017/mlt_voc2'
IGNORE = '###\n'

# 允许 None，因为 None 会在后面检查 IGNORE 时被过滤掉
VALID_LANGUAGE = ['Latin', 'Chinese', 'None']

SCALE = 600
MAX_SCALE_LENGTH = 1200
step = 16

# resize 以后允许的最小高度
MIN_TEXT_HEIGHT = 5

# resize 以后允许的 anchor 数量
MIN_CONTINUE_ANCHORS = 0

# resize 以后允许text line区域最大的高宽比
MAX_HEIGHT_WIDTH_SCALE = 4

# 对文本行进行 split 后，允许最少的 anchor 数量
MIN_ANCHOR_COUNT = 10

global_k_is_none_count = 0


def parse_line(pnts, im_scale):
    """
    :param pnts:
        "x1,y1,x2,y2,x3,y3,x4,y4,language,text"
        矩形四点坐标的顺序： left-top, right-top, right-bottom, left-bottom
    :return:
        (x1,y1,x2,y2,x3,y3,x4,y4), language, text
    """
    splited_line = pnts.split(',')
    if len(splited_line) > 10:
        splited_line[-1] = ','.join(splited_line[10:])

    for i in range(8):
        splited_line[i] = int(int(splited_line[i]) * im_scale)

    pnts = (splited_line[0], splited_line[1],
            splited_line[2], splited_line[3],
            splited_line[4], splited_line[5],
            splited_line[6], splited_line[7])

    return pnts, splited_line[-2], splited_line[-1]


def get_ltrb(line):
    """
    :param line: (x1,y1,x2,y2,x3,y3,x4,y4)
    :return: (xmin, ymin, xmax, ymax)
    """
    xmin = min(line[0], line[6])
    ymin = min(line[1], line[3])
    xmax = max(line[2], line[4])
    ymax = max(line[5], line[7])

    return xmin, ymin, xmax, ymax


def get_img_scale(img, scale, max_scale):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(scale) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_scale:
        im_scale = float(max_scale) / float(im_size_max)

    return im_scale


def clip_line(line, size):
    """
    :param line: (xmin, ymin, xmax, ymax)
    :param size: (height, width)
    :return: (xmin, ymin, xmax, ymax)
    """
    xmin, ymin, xmax, ymax = line

    if xmin < 0:
        xmin = 0
    if xmax > size[1] - 1:
        xmax = size[1] - 1
    if ymin < 0:
        ymin = 0
    if ymax > size[0] - 1:
        ymax = size[0] - 1

    return xmin, ymin, xmax, ymax


def split_text_line(line, step):
    """
    按照 Bounding box 对文字进行划分
    :param line: (xmin, ymin, xmax, ymax)
    :return: [(anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax)]
    """
    xmin, ymin, xmax, ymax = line
    width = xmax - xmin

    anchor_count = int(math.ceil(width / step))

    splited_lines = []

    for i in range(anchor_count):
        anchor_xmin = i * step + xmin
        anchor_ymin = ymin
        anchor_xmax = anchor_xmin + step - 1
        anchor_ymax = ymax

        splited_lines.append((anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax))

    return splited_lines


class Point:
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)

    def __str__(self):
        return "({}, {})".format(self.x, self.y)


class Line:
    def __init__(self, p0: Point, p1: Point):
        self.p0 = p0
        self.p1 = p1

        if p0.x - p1.x == 0:
            self.k = None
        else:
            self.k = (self.p0.y - self.p1.y) / (self.p0.x - self.p1.x)

        # f = ax+by+c = 0
        self.a = self.p0.y - self.p1.y
        self.b = self.p1.x - self.p0.x
        self.c = self.p0.x * self.p1.y - self.p1.x * self.p0.y

    def cross(self, line) -> Point:
        d = self.a * line.b - line.a * self.b
        if d == 0:
            return None

        x = (self.b * line.c - line.b * self.c) / d
        y = (line.a * self.c - self.a * line.c) / d

        return Point(x, y)

    def contain(self, p: Point) -> bool:
        if p is None:
            return False

        # 输入的点应该吃 cross(求出来的交点)
        # p 点是否落在 p0 和 p1 之间, 而不是延长线上
        if p.x > max(self.p1.x, self.p0.x):
            return False

        if p.x < min(self.p1.x, self.p0.x):
            return False

        if p.y > max(self.p1.y, self.p0.y):
            return False

        if p.y < min(self.p1.y, self.p0.y):
            return False

        return True


def get_clockwise(pnts):
    """
    :param pnts: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    :return:
        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        left-top, right-top, right-bottom, left-bottom
    """
    out = []
    p = sorted(pnts, key=lambda x: x[1])
    if p[0][0] < p[1][0]:
        out.append(p[0])
        out.append(p[1])
    else:
        out.append(p[1])
        out.append(p[0])

    if p[2][0] > p[3][0]:
        out.append(p[2])
        out.append(p[3])
    else:
        out.append(p[3])
        out.append(p[2])

    return out


def draw_four_vectors(img, line, color=(0, 255, 0)):
    """
    :param line: (x1,y1,x2,y2,x3,y3,x4,y4)
        矩形四点坐标的顺序： left-top, right-top, right-bottom, left-bottom
    """
    img = cv2.line(img, (line[0], line[1]), (line[2], line[3]), color)
    img = cv2.line(img, (line[2], line[3]), (line[4], line[5]), color)
    img = cv2.line(img, (line[4], line[5]), (line[6], line[7]), color)
    img = cv2.line(img, (line[6], line[7]), (line[0], line[1]), color)
    return img


def draw_bounding_box(img, line, color=(255, 0, 0)):
    """
    :param line: (xmin, ymin, xmax, ymax)
    """
    img = cv2.line(img, (line[0], line[1]), (line[2], line[1]), color)
    img = cv2.line(img, (line[2], line[1]), (line[2], line[3]), color)
    img = cv2.line(img, (line[2], line[3]), (line[0], line[3]), color)
    img = cv2.line(img, (line[0], line[3]), (line[0], line[1]), color)
    return img


def split_text_line2(line, step, img=None):
    """
    按照 minAreaRect 对文本进行划分
    :param line:
        (x1,y1,x2,y2,x3,y3,x4,y4)
        矩形四点坐标的顺序： left-top, right-top, right-bottom, left-bottom
    :return: [(anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax)]
    """
    global global_k_is_none_count
    if DEBUG:
        img = draw_four_vectors(img, line)

    xmin, ymin, xmax, ymax = get_ltrb(line)
    width = xmax - xmin
    height = ymax - ymin

    if height > MAX_HEIGHT_WIDTH_SCALE * width:
        return []

    anchor_count = int(math.ceil(width / step))

    if DEBUG:
        img = draw_bounding_box(img, (xmin, ymin, xmax, ymax))

    rect = cv2.minAreaRect(np.asarray([[line[0], line[1]],
                                       [line[2], line[3]],
                                       [line[4], line[5]],
                                       [line[6], line[7]]]))
    # 获得最小 rotate rect 的四个角点
    box = cv2.boxPoints(rect)
    box = get_clockwise(box)

    if DEBUG:
        img = draw_four_vectors(img, (box[0][0], box[0][1],
                                      box[1][0], box[1][1],
                                      box[2][0], box[2][1],
                                      box[3][0], box[3][1]), color=(255, 55, 55))

    p1 = Point(box[0][0], box[0][1])
    p2 = Point(box[1][0], box[1][1])
    p3 = Point(box[2][0], box[2][1])
    p4 = Point(box[3][0], box[3][1])

    l1 = Line(p1, p2)
    l2 = Line(p2, p3)
    l3 = Line(p3, p4)
    l4 = Line(p4, p1)
    lines = [l1, l2, l3, l4]

    if l1.k is None:
        global_k_is_none_count += 1
        print("l1 K is None")
        print(p1)
        print(p2)
        print(p3)
        print(p4)
        return []

    splited_lines = []
    for i in range(anchor_count):
        anchor_xmin = i * step + xmin
        anchor_xmax = anchor_xmin + step - 1
        anchor_ymin = ymin
        anchor_ymax = ymax

        # 垂直于 X 轴的现
        left_line = Line(Point(anchor_xmin, 0), Point(anchor_xmin, height))
        right_line = Line(Point(anchor_xmax, 0), Point(anchor_xmax, height))

        left_cross_pnts = [left_line.cross(l) for l in lines]
        right_cross_pnts = [right_line.cross(l) for l in lines]

        if l1.k < 0:
            if l1.contain(right_cross_pnts[0]):
                anchor_ymin = right_cross_pnts[0].y

            if l4.contain(right_cross_pnts[3]):
                anchor_ymax = right_cross_pnts[3].y

            if l3.contain(left_cross_pnts[2]):
                anchor_ymax = left_cross_pnts[2].y

            if l2.contain(left_cross_pnts[1]):
                anchor_ymin = left_cross_pnts[1].y

        if l1.k > 0:
            if l4.contain(right_cross_pnts[3]):
                anchor_ymin = right_cross_pnts[3].y

            if l3.contain(right_cross_pnts[2]):
                anchor_ymax = right_cross_pnts[2].y

            if l1.contain(left_cross_pnts[0]):
                anchor_ymin = left_cross_pnts[0].y

            if l2.contain(left_cross_pnts[1]):
                anchor_ymax = left_cross_pnts[1].y

        if anchor_ymax - anchor_ymin <= MIN_TEXT_HEIGHT:
            continue

        if DEBUG:
            img = draw_bounding_box(img, (anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax), (0, 0, 255))
            cv2.imshow('test', img)
            cv2.waitKey()

        splited_lines.append((anchor_xmin, anchor_ymin, anchor_xmax, anchor_ymax))

    if DEBUG:
        cv2.imshow('test', img)
        cv2.waitKey()

    return splited_lines


def test():
    # p1 = Point(5, 1)
    # p2 = Point(1, 4)
    # l1 = Line(p2, p1)
    # print(l1.k)
    #
    # p1 = Point(5, 4)
    # p2 = Point(1, 1)
    # l1 = Line(p1, p2)
    # print(l1.k)

    pnts = [(40, 40), (140, 85), (140, 160), (50, 100)]
    img = np.zeros((200, 200, 3))
    a = cv2.minAreaRect(np.asarray(pnts))

    img = cv2.line(img, pnts[0], pnts[1], (0, 255, 0), thickness=1)
    img = cv2.line(img, pnts[1], pnts[2], (0, 255, 0), thickness=1)
    img = cv2.line(img, pnts[2], pnts[3], (0, 255, 0), thickness=1)
    img = cv2.line(img, pnts[3], pnts[0], (0, 255, 0), thickness=1)

    box = cv2.boxPoints(a)

    def tt(p):
        return (p[0], p[1])

    img = cv2.line(img, tt(box[0]), tt(box[1]), (255, 255, 0), thickness=1)
    img = cv2.line(img, tt(box[1]), tt(box[2]), (255, 255, 0), thickness=1)
    img = cv2.line(img, tt(box[2]), tt(box[3]), (255, 255, 0), thickness=1)
    img = cv2.line(img, tt(box[3]), tt(box[0]), (255, 255, 0), thickness=1)

    cv2.imshow('test', img.astype(np.uint8))
    cv2.waitKey(0)


def main():
    dest_label_dir, dest_img_dir, dest_set_dir = build_voc_dirs(out_dir)
    # training or val
    setname = os.path.basename(img_dir)

    img_names = os.listdir(img_dir)
    img_names.sort()

    trainval_set_file = open(os.path.join(dest_set_dir, setname + '.txt'), 'w')

    count = 0

    # img_names = img_names[0:205]
    for img_name in img_names:
        img_path = os.path.join(img_dir, img_name)

        _, basename = os.path.split(img_name)
        if basename.lower().split('.')[-1] not in ['jpg', 'png']:
            continue

        stem, ext = os.path.splitext(basename)
        gt_file = os.path.join(gt_dir, 'gt_' + stem + '.txt')

        img = cv2.imread(img_path)
        im_scale = get_img_scale(img, SCALE, MAX_SCALE_LENGTH)

        with open(gt_file, 'r') as f:
            lines = f.readlines()

        parsed_lines = [parse_line(line, im_scale) for line in lines]

        if len(parsed_lines) == 0 or \
                not all(x[1] in VALID_LANGUAGE for x in parsed_lines):  # 保证图片中只有 latin/chinese
            print("Skip image: %s" % img_name)
            continue

        parsed_lines = list(filter(lambda x: x[2] != IGNORE, parsed_lines))

        # 过滤掉高度过高或者过小的文本行
        tmp = []
        for l in parsed_lines:
            xmin, ymin, xmax, ymax = get_ltrb(l[0])
            h = ymax - ymin
            w = xmax - xmin
            if MIN_TEXT_HEIGHT < h < MAX_HEIGHT_WIDTH_SCALE * w:
                tmp.append(l)

        parsed_lines = tmp

        resize_img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_AREA)
        resize_img_size = resize_img.shape

        anchors = []
        for line in parsed_lines:
            temp_splited_lines = split_text_line2(line[0], step, resize_img)

            temp_splited_lines = [clip_line(l, resize_img_size) for l in temp_splited_lines]

            if len(temp_splited_lines) <= MIN_CONTINUE_ANCHORS:
                continue

            anchors.extend(temp_splited_lines)

        if len(anchors) <= MIN_ANCHOR_COUNT:
            print("Skip image: %s" % img_name)
            continue

        print("Process image: %s" % img_name)
        out_img_name = '{}_{}.jpg'.format(setname, stem)
        cv2.imwrite(os.path.join(dest_img_dir, out_img_name), resize_img)
        doc, objs = generate_xml(out_img_name, anchors, resize_img_size, 'icdar_mlt17_%s' % setname)

        xmlfile = os.path.join(dest_label_dir, '{}_{}.xml'.format(setname, stem))

        with open(xmlfile, 'w') as f:
            f.write(doc.toprettyxml(indent='	'))

        trainval_set_file.writelines("{}_{}\n".format(setname, stem))
        count += 1

    print("Converted image count: %d" % count)
    print("K is None: %d" % global_k_is_none_count)


if __name__ == "__main__":
    main()
