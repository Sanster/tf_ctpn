import os

from xml.dom.minidom import Document
import numpy as np


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


def generate_xml(img_name, positions, img_size, database, cls='text'):
    """
    :param positions:  [(xmin, ymin, xmax, ymax)]
    """
    doc = Document()

    def append_xml_node_attr(child, parent=None, text=None):
        ele = doc.createElement(child)
        if not text is None:
            text_node = doc.createTextNode(text)
            ele.appendChild(text_node)
        parent = doc if parent is None else parent
        parent.appendChild(ele)
        return ele

    # create header
    annotation = append_xml_node_attr('annotation')
    append_xml_node_attr('folder', parent=annotation, text='text')
    append_xml_node_attr('filename', parent=annotation, text=img_name)
    source = append_xml_node_attr('source', parent=annotation)
    append_xml_node_attr('database', parent=source, text=database)
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
    for pos in positions:
        obj = append_xml_node_attr('object', parent=annotation)
        occlusion = int(0)
        x1, y1, x2, y2 = int(pos[0]), int(pos[1]), int(pos[2]), int(pos[3])

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
