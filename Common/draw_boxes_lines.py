import cv2
import random
import os
from Tfrecord import make_tfrecord
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as mpcm

PIC_CLASS = {
    0: 'Background',
    1: 'tikuan-xz',
    2: 'shantou-t',
    3: 'shantou-j',
    4: 'shanpo-y',
    5: 'fantou',
    6: 'shantou-y',
    7: 'stzh-c',
    8: 'shu-dy',
    9: 'fangwu',
    10: 'yinzhang-xz',
    11: 'shu-cy',
    12: 'stzh-z',
    13: 'shanpo-p',
    14: 'qiao',
    15: 'shu-qy',
    16: 'shanpo-t',
    17: 'yinzhang-qichang',
    18: 'shu-s',
    19: 'shu-ry',
    20: 'shu-xz',
    21: 'tikuan-qc',
    22: 'yinzhang-qc',
    23: 'shantou-p',
    24: 'shu-sy',
    25: 'stzh-cm',
    26: 'stzh-t',
    27: 'shantou-r',
    28: 'chengguan',
    29: 'shantou-h'
}

PIC_CLASS_2 = {
    0: 'none',
    1: 'person',
    2: 'guilty',
    3: 'profile',
}

CLASS_TO_NUM = {
    'none': (0, 'Background'),
    'tikuan-xz': (1, 'tikuan'),
    'shantou-t': (2, 'shantou'),
    'shantou-j': (3, 'shantou'),
    'shanpo-y': (4, 'shanpo'),
    'fantou': (5, 'fantou'),
    'shantou-y': (6, 'shantou'),
    'stzh-c': (7, 'stzh'),
    'shu-dy': (8, 'shu'),
    'fangwu': (9, 'fangwu'),
    'yinzhang-xz': (10, 'yinzhang'),
    'shu-cy': (11, 'shu'),
    'stzh-z': (12, 'stzh'),
    'shanpo-p': (13, 'shanpo'),
    'qiao': (14, 'qiao'),
    'shu-qy': (15, 'shu'),
    'shanpo-t': (16, 'shanpo'),
    'yinzhang-qichang': (17, 'yinzhang'),
    'shu-s': (18, 'shu'),
    'shu-ry': (19, 'shu'),
    'shu-xz': (20, 'shu'),
    'tikuan-qc': (21, 'tikuan'),
    'yinzhang-qc': (22, 'yinzhang'),
    'shantou-p': (23, 'shantou'),
    'shu-sy': (24, 'shu'),
    'stzh-cm': (25, 'stzh'),
    'stzh-t': (26, 'stzh'),
    'shantou-r': (27, 'shantou'),
    'chengguan': (28, 'chengguan'),
    'shantou-h': (29, 'shantou')
}

CLASS_TO_NUM_2 = {
    'none': (0, 'Background'),
    'person': (1, 'person'),
    'guilty': (2, 'guilty'),
    'profile': (3, 'profile'),
}


def colorSelect(colors, num_classes=4):
    dt = len(colors) // num_classes
    sub_colors = []
    for i in range(num_classes):
        color = colors[i * dt]
        if isinstance(color[0], float):
            sub_colors.append([int(c * 255) for c in color])
        else:
            sub_colors.append([c for c in color])
    return sub_colors


colors_plasma = colorSelect(mpcm.plasma.colors, num_classes=4)
colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                  (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                  (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                  (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                  (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]


def drawLines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def drawRect(img, p1, p2, color=[255, 0, 0], thickness=2):
    cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)


def drawBbox(img, bbox, shape, label, color=[255, 0, 0], thickness=2):
    p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
    p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
    cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
    p1 = (p1[0] + 15, p1[1])
    cv2.putText(img, str(label), p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)


def bboxesDrawOnImg(img, classes, scores, bboxes, colors, Frame, thickness=2, dir = PIC_CLASS_2):
    shape = img.shape
    num = '%d' % (Frame[0])
    cv2.putText(img, num, (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        color = colors[classes[i]]
        p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
        p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
        cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
        s = '%s/%.3f' % (dir[classes[i]], scores[i])
        p1 = (p1[0] - 5, p1[1])
        cv2.putText(img, s, p1[::-1], cv2.FONT_HERSHEY_COMPLEX, 0.4, color, 1)


def pltBboxes(img, classes, scores, bboxes, figsize=(10, 10), linewidth=1.5, dir = PIC_CLASS):
    fig = plt.figure(figsize=figsize)
    plt.imshow(img)
    height = img.shape[0]
    width = img.shape[1]
    colors = dict()
    for i in range(classes.shape[0]):
        cls_id = int(classes[i])
        if cls_id >= 0:
            score = scores[i]
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())
            ymin = int(bboxes[i, 0] * height)
            xmin = int(bboxes[i, 1] * width)
            ymax = int(bboxes[i, 2] * height)
            xmax = int(bboxes[i, 3] * width)
            rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                 ymax - ymin, fill=False,
                                 edgecolor=colors[cls_id],
                                 linewidth=linewidth)
            plt.gca().add_patch(rect)
            class_name = str(dir[cls_id])
            plt.gca().text(xmin, ymin - 2,
                           '{:s} | {:.3f}'.format(class_name, score),
                           bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                           fontsize=12, color='white')
    plt.show()


def orignPicWithBboxes(name, path='H://SSD/tmp/',dir=CLASS_TO_NUM_2):
    img, _, bboxes, labels, _, _, _ = make_tfrecord.solveOnePic(name, path, istensor=False, dir=dir)
    return img, bboxes, labels


def classNameToNum(classes, dir = CLASS_TO_NUM):
    class_num = []
    for i in classes:
        class_num.append(dir[i][0])
    return class_num
