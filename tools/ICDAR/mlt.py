#!/usr/bin/env python
from zipfile import ZipFile
import re

GT_TXT_NAME_REG = 'gt_img_([0-9]+).txt'
SUBMIT_TXT_NAME_REG = 'res_img_([0-9]+).txt'


class Validater(object):
    def __init__(self, CRLF=False, LTRB=False):
        """
        :param CRLF: Lines are delimited by Windows CRLF format
        :param LTRB: (left,top,right,bottom) or (x1,y1,x2,y2,x3,y3,x4,y4)
        """
        self.CRLF = CRLF
        self.LTRB = LTRB

    def validate_point_inside_bounds(self, x, y, img_width, img_height):
        if x < 0 or x > img_width:
            raise Exception("X value (%s) not valid. Image dimensions: (%s,%s)" % (x, img_width, img_height))

        if y < 0 or y > img_height:
            raise Exception(
                "Y value (%s)  not valid. Image dimensions: (%s,%s) Sample: %s Line:%s" % (y, img_width, img_height))

    def parse_lines_in_file(self, name, file_contents,
                            with_script=False,
                            with_transcription=False,
                            with_confidence=False,
                            img_width=0,
                            img_height=0):
        lines = file_contents.split('\r\n' if self.CRLF else '\n')
        for line in lines:
            line = line.replace('\r', '').replace('\n', '')
            if line != "":
                try:
                    return self.get_line_values(line, with_script, with_transcription, with_confidence,
                                                img_width, img_height)
                except Exception as e:
                    raise Exception("Line in sample not valid. Sample: %s Line: %s Error: %s" % (name, line, str(e)))

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

    def get_line_values(self, line, with_script=False, with_transcription=False, with_confidence=False, img_width=0,
                        img_height=0):
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

        if self.LTRB:
            true_format = ['xmin', 'ymin', 'xmax', 'ymax']
        else:
            true_format = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']

        if with_confidence:
            optional_patterns.append(confidence_pattern)
            true_format.append('confidence')
        if with_script:
            optional_patterns.append(script_pattern)
            true_format.append('script')
        if with_transcription:
            optional_patterns.append(transcription_pattern)
            true_format.append('transcription')
        true_format = ','.join(true_format)

        def join_parttern(num_points, patterns):
            pattern = ','.join([num_pattern] * num_points)

            for p in patterns:
                pattern += (',%s' % p)

            pattern += '$'

            return pattern

        if self.LTRB:
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

        if with_confidence:
            try:
                confidence = float(m.group(num_points + 1))
            except ValueError:
                raise Exception("Confidence value must be a float")

        if with_script:
            script_pos = num_points + (2 if with_confidence else 1)
            script = m.group(script_pos)
            m2 = re.match(r'^\s*\"(.*)\"\s*$', script)
            if m2 is not None:
                script = m2.group(1)

        if with_transcription:
            if with_confidence and with_script:
                transcription_pos = num_points + 3
            elif with_confidence or with_script:
                transcription_pos = num_points + 2
            else:
                transcription_pos = num_points + 1

            transcription = m.group(transcription_pos)
            m2 = re.match(r'^\s*\"(.*)\"\s*$', transcription)
            if m2 is not None:  # Transcription with double quotes, we extract the value and replace escaped characters
                transcription = m2.group(1).replace("\\\\", "\\").replace("\\\"", "\"")

        return points, confidence, script, transcription


def load_zip_file(file, reg_exp=''):
    """
    :param file: zipFile
    :param reg_exp:
    :return:
        Returns an array with the contents (filtered by reg_exp) of a ZIP file.
        The key's are the names or the file or the capturing group definied in the reg_exp
        The file content are decoded as utf8
    """
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


if __name__ == '__main__':
    gt = load_zip_file('MLT17_val_gt.zip', GT_TXT_NAME_REG)
    submit = load_zip_file('MLT17_val_submit.zip', SUBMIT_TXT_NAME_REG)

    # check whether all samples in submit is in gt
    for k in submit:
        if k not in gt:
            raise Exception("The submit sample %s not present in GT" % k)

    validater = Validater()
    for k, val in gt.items():
        points, confidence, script, transcription = validater.parse_lines_in_file(k, val, with_script=True,
                                                                                  with_transcription=True)
        a = 0
