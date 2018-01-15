import os
import xml.etree.ElementTree as ET

PIC_CLASS = {}
# PIC_CLASS = {
#     'none': (0, 'Background'),
#     'tikuan-xz': (1, 'tikuan'),
#     'shantou-t': (2, 'shantou'),
#     'shantou-j': (3, 'shantou'),
#     'shanpo-y': (4, 'shanpo'),
#     'fantou': (5, 'fantou'),
#     'shantou-y': (6, 'shantou'),
#     'stzh-c': (7, 'stzh'),
#     'shu-dy': (8, 'shu'),
#     'fangwu': (9, 'fangwu'),
#     'yinzhang-xz': (10, 'yinzhang'),
#     'shu-cy': (11, 'shu'),
#     'stzh-z': (12, 'stzh'),
#     'shanpo-p': (13, 'shanpo'),
#     'qiao': (14, 'qiao'),
#     'shu-qy': (15, 'shu'),
#     'shanpo-t': (16, 'shanpo'),
#     'yinzhang-qichang': (17, 'yinzhang'),
#     'shu-s': (18, 'shu'),
#     'shu-ry': (19, 'shu'),
#     'shu-xz': (20, 'shu'),
#     'tikuan-qc': (21, 'tikuan'),
#     'yinzhang-qc': (22, 'yinzhang'),
#     'shantou-p': (23, 'shantou'),
#     'shu-sy': (24, 'shu'),
#     'stzh-cm': (25, 'stzh'),
#     'stzh-t': (26, 'stzh'),
#     'shantou-r': (27, 'shantou'),
#     'chengguan': (28, 'chengguan'),
#     'shantou-h': (29, 'shantou')
# }

def savePicClass(path):
    i = 1
    PicClass = {}
    PicClass.update({'none': (0, 'Background')})
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            if os.path.splitext(filename)[1] == ".xml":
                filepath = os.path.join(dirpath, filename)
                tree = ET.parse(filepath)
                root = tree.getroot()
                for obj in root.findall('object'):
                    temp = {}
                    label = obj.find('name').text
                    if not (label in PicClass.keys()):
                        temp.update({label: (i, label.split('-')[0])})
                        PicClass.update(temp)
                        i += 1
    fclass = open(path + "class.txt", 'w')
    fclass.write(str(PicClass))
    fclass.close()

def getPicClass(path, PicClass = PIC_CLASS):
    savePicClass(path)
    with open(path + "class.txt", "r") as fclass:
        strclass = fclass.read()
        picclass = eval(strclass)
    PicClass.update(picclass)
