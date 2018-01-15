import tensorflow as tf
from Bboxes import bboxes_method
from tensorflow.python.framework import tensor_shape
import random

BBOX_CROP_OVERLAP = 0.7


def imgDistort(image, labels, bboxes, min_object_covered_list, aspect_ratio_range=(0.9, 1.1),
               area_range=(0.1, 1.0), max_attempts=200, isFlip=None, scope=None):
    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bboxes]):
        if isinstance(min_object_covered_list, list):
            min_object_covered = random.choice(min_object_covered_list)
        else:
            min_object_covered = min_object_covered_list
        if min_object_covered > 0:
            bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
                tf.shape(image),
                bounding_boxes=tf.expand_dims(bboxes, 0),
                min_object_covered=min_object_covered,
                aspect_ratio_range=aspect_ratio_range,
                area_range=area_range,
                max_attempts=max_attempts,
                use_image_if_no_bounding_boxes=True)

            distort_bbox = distort_bbox[0, 0]
            cropped_image = tf.slice(image, bbox_begin, bbox_size)
        else:
            cropped_image = image
            distort_bbox = bboxes
        cropped_image.set_shape([None, None, 3])
        bboxes = bboxes_method.bboxesResize(distort_bbox, bboxes)
        labels, bboxes = bboxes_method.bboxesFilter(labels, bboxes, threshold=BBOX_CROP_OVERLAP, isFlip=isFlip)
        return cropped_image, labels, bboxes, distort_bbox


def imgResize(image, size, method=tf.image.ResizeMethod.BILINEAR, align_corners=False):
    with tf.name_scope('resize_image'):
        # height, width, channels = _ImageDimensions(image)
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_images(image, size, method, align_corners)
        image = tf.reshape(image, [size[0], size[1], 3])
        return image


# def imgFlipLR(image, bboxes, seed=None):
#     with tf.name_scope('flip_Left_Right'):
#         image = tf.convert_to_tensor(image, name='image')
#         uniform_random = tf.random_uniform([], 0, 1.0, seed=seed)
#         should_flip = tf.less(uniform_random, 0.5)
#         # result = should_flip ? tf.reverse_v2() : image
#         result = tf.cond(should_flip, lambda: tf.reverse_v2(image, [1]), lambda: image)
#         bboxes = tf.cond(should_flip, lambda: bboxes_method.bboxesFlipLR(bboxes), lambda: bboxes)
#         if image.get_shape() == tensor_shape.unknown_shape():
#             result.set_shape([None, None, None])
#         else:
#             result.set_shape(image.get_shape())
#         return result, bboxes, should_flip

def imgFlipLR(image, bboxes, seed=None):
    with tf.name_scope('flip_Left_Right'):
        image = tf.convert_to_tensor(image, name='image')
        uniform_random = tf.random_uniform([], 0, 1.0, seed=seed)
        should_flip = tf.less(uniform_random, 0.5)
        # result = should_flip ? tf.reverse_v2() : image
        result = tf.cond(should_flip, lambda: tf.image.flip_left_right(image), lambda: image)
        bboxes = tf.cond(should_flip, lambda: bboxes_method.bboxesFlipLR(bboxes), lambda: bboxes)
        if image.get_shape() == tensor_shape.unknown_shape():
            result.set_shape([None, None, None])
        else:
            result.set_shape(image.get_shape())
        return result, bboxes, should_flip


def imgFlipUD(image, bboxes, seed=None):
    with tf.name_scope('flip_Up_Down'):
        image = tf.convert_to_tensor(image, name='image')
        uniform_random = tf.random_uniform([], 0, 1.0, seed=seed)
        should_flip = tf.less(uniform_random, 0.5)
        # result = should_flip ? tf.reverse_v2() : image
        result = tf.cond(should_flip, lambda: tf.image.flip_up_down(image), lambda: image)
        bboxes = tf.cond(should_flip, lambda: bboxes_method.bboxesFlipUD(bboxes), lambda: bboxes)
        if image.get_shape() == tensor_shape.unknown_shape():
            result.set_shape([None, None, None])
        else:
            result.set_shape(image.get_shape())
        return result, bboxes, should_flip


def imgColorDistrot(image, color_ordering=0, scope=None):
    with tf.name_scope(scope, 'distort_color', [image]):
        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
        elif color_ordering == 2:
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        elif color_ordering == 3:
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
        elif color_ordering == 4:
            pass

        return tf.clip_by_value(image, 0.0, 1.0)


def imgWhiten(image):
    # 图像白化，减少相关性，并使特征具有相同的方差
    # mean = tf.constant([123, 117, 104], dtype=image.dtype)  # RGB平均
    # image = image - mean
    tf.image.per_image_standardization(image)
    return image
