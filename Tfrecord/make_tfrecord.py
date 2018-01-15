import os
import sys
import random
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
import matplotlib.image as mpimg
from Data.data_features import float_feature, bytes_feature, int64_feature
from Data.data_fixed import PIC_CLASS
from Data import data_fixed

PIC_SIZE = 300
GLOBAL_PATH = "H://mySSD/PicData/"
MAX_PIC_PER_FILE = 200


def solveOnePic(name, path=GLOBAL_PATH, istensor = True, dir = PIC_CLASS):
    bboxes = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []

    tree = ET.parse(path + name + '.xml')
    root = tree.getroot()

    pic = root.find('filename').text
    filepath = os.path.join(path, pic)
    # im = Image.open(filepath).convert('RGB')
    # out = im.resize((PIC_SIZE, PIC_SIZE), Image.ANTIALIAS)
    # out.save(GLOBAL_PATH + "tmp_jpg/" + name + '.jpg')
    if istensor == True:
        image_data = tf.gfile.FastGFile(filepath, 'rb').read()
    else:
        image_data = 0

    size = root.find('size')
    shape = [int(size.find('height').text),
             int(size.find('width').text),
             int(size.find('depth').text)]

    for obj in root.findall('object'):
        label = obj.find('name').text
        bbox = obj.find('bndbox')

        if obj.find('difficult'):
            difficult.append(int(obj.find('difficult').text))
        else:
            difficult.append(0)

        if obj.find('truncated'):
            truncated.append(int(obj.find('truncated').text))
        else:
            truncated.append(0)

        bboxes.append((float(bbox.find('ymin').text) / shape[0],
                       float(bbox.find('xmin').text) / shape[1],
                       float(bbox.find('ymax').text) / shape[0],
                       float(bbox.find('xmax').text) / shape[1]
                       ))

        labels.append(int(dir[label][0]))
        labels_text.append(label.encode('ascii'))

    return image_data, shape, bboxes, labels, labels_text, difficult, truncated


def convert2Tfrecord(image_data, labels, labels_text, bboxes, shape, difficult, truncated):
    xmin = []
    ymin = []
    xmax = []
    ymax = []

    for b in bboxes:
        assert len(b) == 4
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]

    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(shape[0]),
        'image/width': int64_feature(shape[1]),
        'image/channels': int64_feature(shape[2]),
        'image/shape': int64_feature(shape),
        'bbox/xmin': float_feature(xmin),
        'bbox/xmax': float_feature(xmax),
        'bbox/ymin': float_feature(ymin),
        'bbox/ymax': float_feature(ymax),
        'bbox/label': int64_feature(labels),
        'bbox/label_text': bytes_feature(labels_text),
        'bbox/difficult': int64_feature(difficult),
        'bbox/truncated': int64_feature(truncated),
        'image/format': bytes_feature(image_format),
        'image/encoded': bytes_feature(image_data)}))
    return example


def add2Tfrecord(name, tfrecord_writer, path=GLOBAL_PATH):
    image_data, shape, bboxes, labels, labels_text, difficult, truncated = solveOnePic(name, path)
    example = convert2Tfrecord(image_data, labels, labels_text, bboxes, shape, difficult, truncated)
    tfrecord_writer.write(example.SerializeToString())


def getTfrecord(out_dir, name="train_data", path=GLOBAL_PATH):
    data_fixed.getPicClass(GLOBAL_PATH)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    filelist = os.listdir(GLOBAL_PATH)
    xmllist = []
    for emu in filelist:
        if os.path.splitext(emu)[1] == '.xml':
            xmllist.append(os.path.splitext(emu)[0])
    i = 0
    j = 0
    while i < len(xmllist):
        tf_filename = out_dir + name + ".tfrecord"
        with tf.python_io.TFRecordWriter(tf_filename) as writer:
            while i < len(xmllist):
                add2Tfrecord(xmllist[i], writer, path)
                i += 1
                j += 1
    return xmllist

if __name__ == '__main__':
    getTfrecord("H://mySSD/PicData/")
