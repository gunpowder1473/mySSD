import numpy as np
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import threading
from Bboxes import bboxes_method
from Preprocessing import image_preprocessing
from network import ssd_network
from Common import draw_boxes_lines

CLASS_DIC = {
    'tikuan-xz': [0, 0],
    'shantou-t': [0, 0],
    'shantou-j': [0, 0],
    'shanpo-y': [0, 0],
    'fantou': [0, 0],
    'shantou-y': [0, 0],
    'stzh-c': [0, 0],
    'shu-dy': [0, 0],
    'fangwu': [0, 0],
    'yinzhang-xz': [0, 0],
    'shu-cy': [0, 0],
    'stzh-z': [0, 0],
    'shanpo-p': [0, 0],
    'qiao': [0, 0],
    'shu-qy': [0, 0],
    'shanpo-t': [0, 0],
    'yinzhang-qichang': [0, 0],
    'shu-s': [0, 0],
    'shu-ry': [0, 0],
    'shu-xz': [0, 0],
    'tikuan-qc': [0, 0],
    'yinzhang-qc': [0, 0],
    'shantou-p': [0, 0],
    'shu-sy': [0, 0],
    'stzh-cm': [0, 0],
    'stzh-t': [0, 0],
    'shantou-r': [0, 0],
    'chengguan': [0, 0],
    'shantou-h': [0, 0],
}

CLASS_DIC_2 = {
    'person': [0, 0],
    'profile': [0, 0],
    'guilty': [0, 0]
}

slim = tf.contrib.slim

# 构建会话
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)
net_shape = (300, 300)

# 数据格式与存储结构
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))

# SSD网络尺寸分配.
image_pre, labels_pre, bboxes_pre, bbox_img = image_preprocessing.preprocessEval(img_input, None, None, net_shape,
                                                                                 data_format)

# image_4d = [1,image_pre]
image_4d = tf.expand_dims(image_pre, 0)

# 定义SSD模型
# 查找是否定义ssd_net变量
reuse = True if 'ssd_net' in locals() else None
ssd_params = ssd_network.SSDNet_300.default_params._replace(num_classes=4)
ssd_net = ssd_network.SSDNet_300(ssd_params)

# slim的默认ssd网络的默认数据格式是NHWC
with slim.arg_scope(ssd_net.argScope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.creatNetwork(image_4d, is_training=False, reuse=reuse,
                                                            scope='ssd_300')

# 调用训练好的SSD模型.
ckpt_filename = 'H://mySSD/checkpoint2/model4.ckpt'  # 'E://SSD/checkpoints/ssd_300_vgg.ckpt/ssd_300_vgg.ckpt'
# 初始化全部张量
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.creatDefaultbox(net_shape)

curFrame = [0, False]


def processImage(img, select_threshold=.5, nms_threshold=.45):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})

    rclasses, rscores, rbboxes = bboxes_method.npBboxesSelectALLLayers(rpredictions, rlocalisations, ssd_anchors,
                                                                       select_threshold=select_threshold)

    rbboxes = bboxes_method.npBboxesLimit(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = bboxes_method.npBboxesSorted(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = bboxes_method.npBboxesNms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    rbboxes = bboxes_method.npBboxesResize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes


def calcAccuracy(bboxesref, bboxes, classesref, classes, thresold=.5, dir1=CLASS_DIC,
                 dir2=draw_boxes_lines.PIC_CLASS):
    bboxesref = np.asarray(bboxesref)
    bboxes = np.asarray(bboxes)
    classesref = np.asarray(classesref)
    classes = np.asarray(classes)
    total_bboxes = len(bboxesref)
    wrong_bboxes = 0
    wrong_classes = []
    for i, s in enumerate(bboxesref):
        dir1[dir2[classesref[i]]][1] += 1
        result = bboxes_method.npBboxesAccuracy(s, bboxes, classesref[i], classes, thresold)
        if result is False:
            wrong_bboxes += 1
            wrong_classes.append(classesref[i])
            dir1[dir2[classesref[i]]][0] += 1
    return (1 - wrong_bboxes / total_bboxes) * 100, wrong_bboxes, total_bboxes, wrong_classes


def solve(path, dir1=CLASS_DIC, dir2=draw_boxes_lines.PIC_CLASS):
    result = []
    wrong = []
    total_num = []
    wrong_num = []
    filelist = os.listdir(path)
    for enum in filelist:
        if os.path.splitext(enum)[1] == '.xml':
            name = os.path.splitext(enum)[0]
            img = cv2.imread(path + name + ".jpg", 1)
            if img.ndim != 3:
                pass
            rclasses, rscores, rbboxes = processImage(img)
            _, bboxesref, classesref = draw_boxes_lines.orignPicWithBboxes(name, path, draw_boxes_lines.CLASS_TO_NUM)
            accuracy, _, total, cur_wrong = calcAccuracy(bboxesref, rbboxes, classesref, rclasses, dir1=dir1,
                                                         dir2=dir2, thresold=.5)
            cur_wrong = [dir2[i] for i in cur_wrong]
            wrong.append(cur_wrong)
            wrong_num.append(len(cur_wrong))
            print("Picture named " + name + "'s accuracy is {:.3f} %, with the wrong classes are {}!"
                  .format(accuracy, cur_wrong))
            result.append(accuracy)
            total_num.append(total)
            if 0 == accuracy:
                draw_boxes_lines.pltBboxes(img, rclasses, rscores, rbboxes, dir=dir2)
                draw_boxes_lines.pltBboxes(img, np.asarray(classesref), np.ones_like(np.asanyarray(classesref)),
                                           np.asarray(bboxesref), dir=dir2)
    result = np.asarray(result)
    wrong_num = np.asarray(wrong_num)
    total_num = np.asarray(total_num)
    return result.mean(), result.max(), result.min(), wrong, (1 - wrong_num.sum() / total_num.sum()) * 100


def solveV2(name, path='H://mySSD/tmp'):
    img = cv2.imread(path + name + ".jpg", 1)
    if img.ndim != 3:
        pass
    rclasses, rscores, rbboxes = processImage(img)
    _, bboxesref, classesref = draw_boxes_lines.orignPicWithBboxes(name)
    draw_boxes_lines.pltBboxes(img, rclasses, rscores, rbboxes)
    draw_boxes_lines.pltBboxes(img, np.asarray(classesref), np.ones_like(np.asanyarray(classesref)),
                               np.asarray(bboxesref))


def solveV3(name, path='H://mySSD/tmp'):
    img = cv2.imread(path + name + ".jpg", 1)
    if img.ndim != 3:
        pass
    rclasses, rscores, rbboxes = processImage(img)
    _, bboxesref, classesref = draw_boxes_lines.orignPicWithBboxes(name)
    accuracy = calcAccuracy(bboxesref, rbboxes, classesref, rclasses)
    print("Picture named " + name + "'s accuracy is {:.3f} %!".format(accuracy))
    if accuracy < 1:
        draw_boxes_lines.pltBboxes(img, rclasses, rscores, rbboxes)
        draw_boxes_lines.pltBboxes(img, np.asarray(classesref), np.ones_like(np.asanyarray(classesref)),
                                   np.asarray(bboxesref))


def solveV4(path):
    cam = cv2.VideoCapture(path)
    success, img = cam.read()
    Frame = 0
    while success:
        rclasses, rscores, rbboxes = processImage(img)
        Frame += 1
        if curFrame[1] is True:
            curFrame[0] = Frame
            Frame = 0
            curFrame[1] = False
        draw_boxes_lines.bboxesDrawOnImg(img, rclasses, rscores, rbboxes, draw_boxes_lines.colors_plasma,
                                         curFrame)
        cv2.imshow('test', img)
        c = cv2.waitKey(10)
        if c == 27:
            break
        success, img = cam.read()


def creatThreadforTimer():
    curFrame[1] = True
    t = threading.Timer(1, creatThreadforTimer)
    t.start()


if __name__ == '__main__':
    creatThreadforTimer()
    solveV4(0)
    # mean, max, min, _, acc = solve('H://mySSD/PicData/',CLASS_DIC, draw_boxes_lines.PIC_CLASS)
    # print("The mean accuracy is {:.3f}%, The max accuracy is {:.3f}%, The min accuracy is {:.3f}%.\r\n "
    #       .format(acc, max, min))
    # for key in CLASS_DIC:
    #     if CLASS_DIC[key][1] is not 0:
    #         acy = (1 - CLASS_DIC[key][0] / CLASS_DIC[key][1]) * 100
    #         print("The class {:s} has appered {:d} times, and {:d} times are wrong, it's accuracy is {:.3f}%"
    #             .format(key, CLASS_DIC[key][1], CLASS_DIC[key][0], acy))
