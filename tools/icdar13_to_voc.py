from xml.dom.minidom import Document
import cv2
import os
import glob
import shutil
import numpy as np


def generate_xml(img_name, lines, img_size, class_sets):
    doc = Document()

    def append_xml_node_attr(child, parent=None, text=None):
        ele = doc.createElement(child)
        if not text is None:
            text_node = doc.createTextNode(text)
            ele.appendChild(text_node)
        parent = doc if parent is None else parent
        parent.appendChild(ele)
        return ele

    cls = 'text'

    # create header
    annotation = append_xml_node_attr('annotation')
    append_xml_node_attr('folder', parent=annotation, text='text')
    append_xml_node_attr('filename', parent=annotation, text=img_name)
    source = append_xml_node_attr('source', parent=annotation)
    append_xml_node_attr('database', parent=source, text='coco_text_database')
    append_xml_node_attr('annotation', parent=source, text='text')
    append_xml_node_attr('image', parent=source, text='text')
    append_xml_node_attr('flickrid', parent=source, text='000000')
    owner = append_xml_node_attr('owner', parent=annotation)
    append_xml_node_attr('name', parent=owner, text='ms')
    size = append_xml_node_attr('size', annotation)
    append_xml_node_attr('width', size, str(img_size[1]))
    append_xml_node_attr('height', size, str(img_size[0]))
    append_xml_node_attr('depth', size, str(img_size[2]))
    append_xml_node_attr('segmented', parent=annotation, text='0')

    # create objects
    objs = []
    for line in lines:
        splitted_line = line.strip().lower().split()

        obj = append_xml_node_attr('object', parent=annotation)
        occlusion = int(0)
        x1, y1, x2, y2 = int(float(splitted_line[0])), int(float(splitted_line[1])), \
                         int(float(splitted_line[2])), int(float(splitted_line[3]))

        truncation = float(0)
        difficult = 0
        truncted = 0 if truncation < 0.5 else 1

        append_xml_node_attr('name', parent=obj, text=cls)
        append_xml_node_attr('pose', parent=obj, text='none')
        append_xml_node_attr('truncated', parent=obj, text=str(truncted))
        append_xml_node_attr('difficult', parent=obj, text=str(int(difficult)))
        bb = append_xml_node_attr('bndbox', parent=obj)
        append_xml_node_attr('xmin', parent=bb, text=str(x1))
        append_xml_node_attr('ymin', parent=bb, text=str(y1))
        append_xml_node_attr('xmax', parent=bb, text=str(x2))
        append_xml_node_attr('ymax', parent=bb, text=str(y2))

        o = {'class': cls, 'box': np.asarray([x1, y1, x2, y2], dtype=float),
             'truncation': truncation, 'difficult': difficult, 'occlusion': occlusion}
        objs.append(o)

    return doc, objs


def _is_hard(cls, truncation, occlusion, x1, y1, x2, y2):
    hard = False
    if y2 - y1 < 25 and occlusion >= 2:
        hard = True
        return hard
    if occlusion >= 3:
        hard = True
        return hard
    if truncation > 0.8:
        hard = True
        return hard
    return hard


def build_voc_dirs(outdir):
    mkdir = lambda dir: os.makedirs(dir) if not os.path.exists(dir) else None
    mkdir(outdir)
    mkdir(os.path.join(outdir, 'Annotations'))
    mkdir(os.path.join(outdir, 'ImageSets'))
    mkdir(os.path.join(outdir, 'ImageSets', 'Layout'))
    mkdir(os.path.join(outdir, 'ImageSets', 'Main'))
    mkdir(os.path.join(outdir, 'ImageSets', 'Segmentation'))
    mkdir(os.path.join(outdir, 'JPEGImages'))
    mkdir(os.path.join(outdir, 'SegmentationClass'))
    mkdir(os.path.join(outdir, 'SegmentationObject'))
    return os.path.join(outdir, 'Annotations'), os.path.join(outdir, 'JPEGImages'), os.path.join(outdir, 'ImageSets',
                                                                                                 'Main')


if __name__ == '__main__':
    outdir = '/home/cwq/data/ICDAR13/icdar13_voc'
    dest_label_dir, dest_img_dir, dest_set_dir = build_voc_dirs(outdir)

    for dset in ['train']:
        _labeldir = '/home/cwq/data/ICDAR13/Challenge2_Training_Task1_GT_splited'
        _imagedir = '/home/cwq/data/ICDAR13/Challenge2_Training_Task12_Images_splited'
        class_sets = ('text', 'dontcare')
        class_sets_dict = dict((k, i) for i, k in enumerate(class_sets))
        fs = [open(os.path.join(dest_set_dir, cls + '_' + dset + '.txt'), 'w') for cls in class_sets]
        ftrain = open(os.path.join(dest_set_dir, dset + '.txt'), 'w')

        files = glob.glob(os.path.join(_labeldir, '*.txt'))
        files.sort()
        for file in files:
            path, basename = os.path.split(file)
            stem, ext = os.path.splitext(basename)
            img_id = stem.split('_')[1]
            img_name = img_id + '.jpg'
            stem = "icdar13_" + img_id
            with open(file, 'r') as f:
                lines = f.readlines()
            img_file = os.path.join(_imagedir, img_name)

            print(img_file)
            img = cv2.imread(img_file)
            img_size = img.shape

            save_img_name = "icdar13_" + img_name
            doc, objs = generate_xml(save_img_name, lines, img_size, class_sets=class_sets)

            cv2.imwrite(os.path.join(dest_img_dir, save_img_name), img)
            xmlfile = os.path.join(dest_label_dir, stem + '.xml')

            with open(xmlfile, 'w') as f:
                f.write(doc.toprettyxml(indent='	'))

            ftrain.writelines(stem + '\n')

        (f.close() for f in fs)
        ftrain.close()
