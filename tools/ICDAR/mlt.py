#!/usr/bin/env python
import argparse
import time
from functools import reduce
from zipfile import ZipFile
from collections import namedtuple
import numpy as np
import Polygon as plg
import re
import os

DEBUG = True
GT_TXT_NAME_REG = 'gt_img_([0-9]+).txt'
SUBMIT_TXT_NAME_REG = 'res_img_([0-9]+).txt'
DONT_CARE = "###"
AREA_PRECISION_CONSTRAINT = 0.5
IOU_CONSTRAINT = 0.5

LineFormat = namedtuple('LineFormat', ['crlf',  # Line is delimited by windows CRLF format
                                       'ltrb',  # 2 points (left,top,right,bottom) or 4 points (x1,y1,x2,y2,x3,y3,x4,y4)
                                       # Points [,confidence][,script][,transcription]
                                       'with_confidence',  # line include confidence
                                       'with_script',  # line include script
                                       'with_transcription'  # line include transcription
                                       ])


class Validater(object):
    def validate_point_inside_bounds(self, x, y, img_width, img_height):
        if x < 0 or x > img_width:
            raise Exception("X value (%s) not valid. Image dimensions: (%s,%s)" % (x, img_width, img_height))

        if y < 0 or y > img_height:
            raise Exception(
                "Y value (%s)  not valid. Image dimensions: (%s,%s) Sample: %s Line:%s" % (y, img_width, img_height))

    def parse_lines_in_file(self, name, file_contents, line_format, img_width=0, img_height=0):
        points_list = []
        confidence_list = []
        script_list = []
        transcription_list = []
        lines = file_contents.split('\r\n' if line_format.crlf else '\n')
        for line in lines:
            line = line.replace('\r', '').replace('\n', '')
            if line != "":
                try:
                    points, confidence, script, transcription = self.get_line_values(line, line_format, img_width,
                                                                                     img_height)
                    points_list.append(points)
                    confidence_list.append(confidence)
                    script_list.append(script)
                    transcription_list.append(transcription)
                except Exception as e:
                    raise Exception("Line in sample not valid. Sample: %s Line: %s Error: %s" % (name, line, str(e)))

        return points_list, confidence_list, script_list, transcription_list

    def validate_clockwise_points(self, points):
        """
        Validates that the points that the 4 points that dlimite a polygon are in clockwise order.
        """

        if len(points) != 8:
            raise Exception("Points list not valid." + str(len(points)))

        point = [
            [int(points[0]), int(points[1])],
            [int(points[2]), int(points[3])],
            [int(points[4]), int(points[5])],
            [int(points[6]), int(points[7])]
        ]
        edge = [
            (point[1][0] - point[0][0]) * (point[1][1] + point[0][1]),
            (point[2][0] - point[1][0]) * (point[2][1] + point[1][1]),
            (point[3][0] - point[2][0]) * (point[3][1] + point[2][1]),
            (point[0][0] - point[3][0]) * (point[0][1] + point[3][1])
        ]

        summatory = edge[0] + edge[1] + edge[2] + edge[3]
        if summatory > 0:
            raise Exception("Points are not clockwise. The coordinates of bounding quadrilaterals "
                            "have to be given in clockwise order. Regarding the correct interpretation of "
                            "'clockwise' remember that the image coordinate system used is the standard one, "
                            "with the image origin at the upper left, the X axis extending to the right and Y "
                            "axis extending downwards.")

    def get_line_values(self, line, line_format, img_width=0, img_height=0):
        """
        Returns values from a textline. Points , [Confidences], [script], [Transcriptions]
        Posible values are:
        LTRB=True: xmin,ymin,xmax,ymax[,confidence][,script][,transcription]
        LTRB=False: x1,y1,x2,y2,x3,y3,x4,y4[,confidence][,script][,transcription]
        """
        confidence = 0.0
        transcription = ""
        script = ""
        points = []

        num_pattern = r'\s*(-?[0-9]+)\s*'
        confidence_pattern = r'\s*([0-1].?[0-9]*)\s*'
        transcription_pattern = r'(.*)'
        script_pattern = r'([a-zA-Z]+)'

        optional_patterns = []

        if line_format.ltrb:
            true_format = ['xmin', 'ymin', 'xmax', 'ymax']
        else:
            true_format = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']

        if line_format.with_confidence:
            optional_patterns.append(confidence_pattern)
            true_format.append('confidence')
        if line_format.with_script:
            optional_patterns.append(script_pattern)
            true_format.append('script')
        if line_format.with_transcription:
            optional_patterns.append(transcription_pattern)
            true_format.append('transcription')
        true_format = ','.join(true_format)

        def join_parttern(num_points, patterns):
            pattern = ','.join([num_pattern] * num_points)

            for p in patterns:
                pattern += (',%s' % p)

            pattern += '$'

            return pattern

        if line_format.ltrb:
            num_points = 4
            m = re.match(join_parttern(num_points, optional_patterns), line)
            if m is None:
                raise Exception("Format incorrect. Should be: %s" % true_format)

            xmin = int(m.group(1))
            ymin = int(m.group(2))
            xmax = int(m.group(3))
            ymax = int(m.group(4))
            if xmax < xmin:
                raise Exception("Xmax value (%s) not valid (Xmax < Xmin)." % xmax)
            if ymax < ymin:
                raise Exception("Ymax value (%s)  not valid (Ymax < Ymin)." % ymax)

            points = [float(m.group(i)) for i in range(1, (num_points + 1))]

            if img_width > 0 and img_height > 0:
                self.validate_point_inside_bounds(xmin, ymin, img_width, img_height)
                self.validate_point_inside_bounds(xmax, ymax, img_width, img_height)

        else:
            num_points = 8
            m = re.match(join_parttern(num_points, optional_patterns), line)
            if m is None:
                raise Exception("Format incorrect. Should be: %s" % true_format)

            points = [float(m.group(i)) for i in range(1, (num_points + 1))]

            self.validate_clockwise_points(points)

            if img_width > 0 and img_height > 0:
                self.validate_point_inside_bounds(points[0], points[1], img_width, img_height)
                self.validate_point_inside_bounds(points[2], points[3], img_width, img_height)
                self.validate_point_inside_bounds(points[4], points[5], img_width, img_height)
                self.validate_point_inside_bounds(points[6], points[7], img_width, img_height)

        if line_format.with_confidence:
            try:
                confidence = float(m.group(num_points + 1))
            except ValueError:
                raise Exception("Confidence value must be a float")

        if line_format.with_script:
            script_pos = num_points + (2 if line_format.with_confidence else 1)
            script = m.group(script_pos)
            m2 = re.match(r'^\s*\"(.*)\"\s*$', script)
            if m2 is not None:
                script = m2.group(1)

        if line_format.with_transcription:
            if line_format.with_confidence and line_format.with_script:
                transcription_pos = num_points + 3
            elif line_format.with_confidence or line_format.with_script:
                transcription_pos = num_points + 2
            else:
                transcription_pos = num_points + 1

            transcription = m.group(transcription_pos)
            m2 = re.match(r'^\s*\"(.*)\"\s*$', transcription)
            if m2 is not None:  # Transcription with double quotes, we extract the value and replace escaped characters
                transcription = m2.group(1).replace("\\\\", "\\").replace("\\\"", "\"")

        return points, confidence, script, transcription


# For gt data, if transcription == '###', dont_care = true
# For submit data, if overlapping more than 50% with “do not care” ground truth regions, dont_care = true
LineData = namedtuple('LineData', ['points', 'pol', 'confidence', 'script', 'transcription', 'dont_care'])


class Evaluater(object):
    def __init__(self):
        self.validater = Validater()

    def polygon_from_ltrb(self, rect):
        rect = [int(x) for x in rect]
        points = [
            (rect[0], rect[1]),
            (rect[2], rect[1]),
            (rect[2], rect[3]),
            (rect[0], rect[3])
        ]

        return plg.Polygon(points)

    def polygon_from_nonltrb(self, points):
        points = [int(x) for x in points]
        points = [
            (points[0], points[1]),
            (points[2], points[3]),
            (points[4], points[5]),
            (points[6], points[7])
        ]
        return plg.Polygon(points)

    def process(self, zip_data, line_format):
        out = {}
        for name, file_content in zip_data.items():
            points_list, confidence_list, script_list, transcription_list = \
                self.validater.parse_lines_in_file(name,
                                                   file_content,
                                                   line_format)

            file_line_data = []
            for i, points in enumerate(points_list):
                if line_format.ltrb:
                    pol = self.polygon_from_ltrb(points)
                else:
                    pol = self.polygon_from_nonltrb(points)

                data = LineData(points, pol, confidence_list[i], script_list[i], transcription_list[i],
                                transcription_list[i] == DONT_CARE)
                file_line_data.append(data)

            out[name] = file_line_data

        return out

    def get_pol_intersection(self, pol1, pol2):
        intersection_pol = pol1 & pol2
        if len(intersection_pol) == 0:
            return 0
        return intersection_pol.area()

    def get_union(self, pol1, pol2):
        area1 = pol1.area()
        area2 = pol2.area()
        return area1 + area2 - self.get_pol_intersection(pol1, pol2)

    def get_pol_iou(self, pol1, pol2):
        try:
            return self.get_pol_intersection(pol1, pol2) / self.get_union(pol1, pol2)
        except:
            return 0

    # Any detections overlapping more than 50% with “do not care” ground truth regions will be discarded
    def mark_sub_dont_care(self, sub_line_datas, gt_line_datas):
        out = []
        for sub_line in sub_line_datas:
            for gt_line in gt_line_datas:
                if gt_line.dont_care:
                    intersected_area = self.get_pol_intersection(sub_line.pol, gt_line.pol)
                    sub_area = sub_line.pol.area()
                    overlap = 0 if sub_area == 0 else intersected_area / sub_area
                    if overlap > AREA_PRECISION_CONSTRAINT:
                        new_sub_line = LineData(sub_line.points, sub_line.pol, sub_line.confidence, sub_line.script,
                                                sub_line.transcription, True)
                        out.append(new_sub_line)
                        break
            else:
                out.append(sub_line)
        return out

    def hmean(self, recall, precision):
        hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)
        return hmean

    def eval(self, gt_filepath, sub_filepath, gt_line_format, sub_line_format):
        gt = load_zip_file(gt_filepath, GT_TXT_NAME_REG)
        submit = load_zip_file(sub_filepath, SUBMIT_TXT_NAME_REG)

        if DEBUG:
            print('gt img count: %d, sub img count: %d' % (len(gt), len(submit)))

        # check whether all samples in submit is in gt
        for k in submit:
            if k not in gt:
                raise Exception("The submit sample %s not present in GT" % k)

        # key: img file id, e.g img_123.jpg, id=123
        # value: LineData list of each img file
        gt_datas = self.process(gt, gt_line_format)

        sub_datas = self.process(submit, sub_line_format)

        # Filter sub_datas's img_id not in gt_datas
        sub_datas = {key: sub_datas[key] for key in gt_datas.keys() if key in sub_datas.keys()}

        sub_datas = {key: self.mark_sub_dont_care(sub_datas[key], gt_datas[key]) for key in gt_datas.keys()}

        if DEBUG:
            print('gt img data count: %d, sub img data count: %d' % (len(gt_datas), len(sub_datas)))

        # key: image id
        # value: matched index of gt line and sub line in one image
        match_pairs = {}
        matrix = {}
        num_sub_matched_total = 0
        num_gt_care_total = 0
        num_sub_care_total = 0
        for img_id, gt_line_datas in gt_datas.items():
            log = 'img_%s\n' % img_id
            sub_line_datas = sub_datas[img_id]
            iou_mat = np.empty([len(gt_line_datas), len(sub_line_datas)])
            gt_matched = [False] * len(gt_line_datas)
            sub_matched = [False] * len(sub_line_datas)

            num_gt_care = reduce(lambda x, y: x if y.dont_care else x + 1, gt_line_datas, 0)
            num_sub_care = reduce(lambda x, y: x if y.dont_care else x + 1, sub_line_datas, 0)

            log += 'GT rects: %d (%d don\'t care)\n' % (len(gt_line_datas), len(gt_line_datas) - num_gt_care)
            log += 'SUB rects: %d (%d don\'t care)\n' % (len(sub_line_datas), len(sub_line_datas) - num_sub_care)

            num_sub_matched = 0

            for gi, gt_line in enumerate(gt_line_datas):
                for si, sub_line in enumerate(sub_line_datas):
                    iou = self.get_pol_iou(gt_line.pol, sub_line.pol)
                    iou_mat[gi, si] = iou

                    if not gt_line.dont_care and not sub_line.dont_care \
                            and not gt_matched[gi] and not sub_matched[si]:
                        if iou > IOU_CONSTRAINT:
                            gt_matched[gi] = True
                            sub_matched[si] = True
                            num_sub_matched += 1
                            match_pairs[img_id] = {'gt': gi, 'sub': si}
                            log += 'Match GT #%d with SUB #%d\n' % (gi, si)

            if num_gt_care == 0:
                recall = 1.0
                precision = 0.0 if num_sub_care > 0 else 1.0
            else:
                recall = num_sub_matched / num_gt_care
                precision = 0 if num_sub_care == 0 else num_sub_matched / num_sub_care

            hmean = self.hmean(recall, precision)

            num_sub_matched_total += num_sub_matched
            num_gt_care_total += num_gt_care
            num_sub_care_total += num_sub_care

            # TODO save per sample results
            print(log)

        # TODO cal MAP MAR

        total_recall = 0 if num_gt_care_total == 0 else num_sub_matched_total / num_gt_care_total
        total_precision = 0 if num_sub_care_total == 0 else num_sub_matched_total / num_sub_care_total
        total_hmean = self.hmean(total_recall, total_precision)

        if DEBUG:
            print("num gt care: %d" % num_gt_care_total)
            print("num sub care: %d" % num_sub_care_total)
            print("num sub matched: %d" % num_sub_matched_total)

        # Print result
        print("Recall: %.04f" % total_recall)
        print("Precision: %.04f" % total_precision)
        print("hmean: %.04f" % total_hmean)


def load_zip_file(file, reg_exp=''):
    """
    :param file: zipFile
    :param reg_exp:
    :return:
        Returns an array with the contents (filtered by reg_exp) of a ZIP file.
        The key's are the names or the file or the capturing group definied in the reg_exp
        The file content are decoded as utf8
    """
    if not os.path.exists(file):
        print("Zipfile %s not exists" % file)
        exit(-1)
    try:
        archive = ZipFile(file, mode='r', allowZip64=True)
    except:
        raise Exception('Error loading the ZIP archive')

    pairs = {}
    for name in archive.namelist():
        isvalid = True
        key = name
        if reg_exp != "":
            m = re.match(reg_exp, name)
            if m is None:
                isvalid = False
            else:
                if len(m.groups()) > 0:
                    key = m.group(1)

        if isvalid:
            pairs[key] = archive.read(name).decode('utf8')
        else:
            raise Exception('ZIP entry not valid: %s' % name)

    return pairs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, default='./MLT17_val_gt.zip')
    parser.add_argument('--sub', type=str, default='./MLT17_submit_fc.zip')
    parser.add_argument('--challenge', type=str, choices=['ICDAR13', 'ICDAR15', 'MLT'])
    args = parser.parse_args()

    if not os.path.exists(args.gt):
        parser.error("%s not exists" % args.gt)

    if not os.path.exists(args.sub):
        parser.error("%s not exists" % args.sub)

    return args


if __name__ == '__main__':
    args = parse_args()

    if args.challenge == 'MLT':
        gt_line_format = LineFormat(crlf=False, ltrb=False,
                                    with_confidence=False,
                                    with_script=True,
                                    with_transcription=True)

        sub_line_format = LineFormat(crlf=False, ltrb=False,
                                     with_confidence=False,
                                     with_script=False,
                                     with_transcription=False)
    elif args.challenge == 'ICDAR13':
        # TODO: add ICDAR13 eval method
        global AREA_PRECISION_CONSTRAINT
        AREA_PRECISION_CONSTRAINT = 0.4  # Same as ICDAR13 script.py
        gt_line_format = LineFormat(crlf=False, ltrb=True,
                                    with_confidence=False,
                                    with_script=False,
                                    with_transcription=True)

        sub_line_format = LineFormat(crlf=False, ltrb=True,
                                     with_confidence=False,
                                     with_script=False,
                                     with_transcription=False)

    start_time = time.time()
    e = Evaluater()
    e.eval(args.gt, args.sub, gt_line_format, sub_line_format)
    print("\n Evaluate time: %f s" % (time.time() - start_time))
