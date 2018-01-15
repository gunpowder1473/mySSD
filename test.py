import numpy as np
import cv2


def chrToNum(chr):
    if '9' >= chr >= '0':
        return ord(chr) - ord('0')
    elif 'A' <= chr <= 'F':
        return ord(chr) - ord('A') + 10


def decodeJPG(path="H://zj_pic.txt"):
    string = open(path).read()
    string = string.split(" ")
    toarray = np.asarray(string)
    result = np.zeros_like(toarray, dtype=np.uint8)
    for i, s in enumerate(toarray):
        temp1 = chrToNum(s[0])
        temp2 = chrToNum(s[1])
        result[i] = (temp1 << 4) + temp2
    return result


if __name__ == '__main__':
    array = decodeJPG()
    img_decode = cv2.imdecode(array, cv2.IMWRITE_JPEG_QUALITY)
    cv2.imshow('test', img_decode)
    cv2.waitKey(0)
