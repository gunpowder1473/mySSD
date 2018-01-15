import tensorflow as tf
from Imagine import image_method
from tensorflow.python.ops import control_flow_ops

MIN_OBJECT_COVERED_LIST = [-1, -1, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
CROP_RATIO_RANGE = (0.5, 2)


def preprocessImage(image, labels, bboxes, out_shape, data_format, is_training=False, **kwargs):
    if is_training:
        return preprocessTrain(image, labels, bboxes, out_shape=out_shape, data_format=data_format)
    else:
        return preprocessEval(image, labels, bboxes, out_shape=out_shape, data_format=data_format, **kwargs)


def preprocessTrain(image, labels, bboxes, out_shape, data_format='NHWC', scope='ssd_preprocessing_train'):
    with tf.name_scope(scope, 'ssd_preprocessing_train', [image, labels, bboxes]):

        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)  # [0,1]
            summaryImage(image, bboxes, 'image_with_bboxes')

        dst_image, bboxes, should_flip1 = image_method.imgFlipLR(image, bboxes)
        dst_image, bboxes, should_flip2 = image_method.imgFlipUD(dst_image, bboxes)
        should_flip = tf.logical_or(should_flip1, should_flip2)

        dst_image, labels, bboxes, distort_bbox = \
            image_method.imgDistort(dst_image, labels, bboxes, min_object_covered_list=MIN_OBJECT_COVERED_LIST,
                                    aspect_ratio_range=CROP_RATIO_RANGE, isFlip=should_flip)
        dst_image = image_method.imgResize(dst_image, out_shape, method=tf.image.ResizeMethod.BILINEAR,
                                           align_corners=False)
        summaryImage(dst_image, bboxes, 'image_shape_distorted')

        dst_image = randomSelect(dst_image, lambda x, ordering: image_method.imgColorDistrot(x, ordering), num_cases=5)
        summaryImage(dst_image, bboxes, 'image_color_distorted')

        dst_image = image_method.imgWhiten(dst_image)
        summaryImage(dst_image, bboxes, 'image_whiten')

        image = dst_image * 255.
        if data_format == 'NCHW':
            image = tf.transpose(image, perm=(2, 0, 1))
        return image, labels, bboxes


def preprocessEval(image, labels, bboxes, out_shape=(300, 300), data_format='NHWC', scope='ssd_preprocessing_eval'):
    with tf.name_scope(scope):
        image = tf.to_float(image)
        image = image_method.imgWhiten(image)
        bbox_img = tf.constant([[0., 0., 1., 1.]])
        if bboxes is None:
            bboxes = bbox_img
        else:
            bboxes = tf.concat([bbox_img, bboxes], axis=0)
        image = image_method.imgResize(image, out_shape, method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
        bbox_img = bboxes[0]
        bboxes = bboxes[1:]
        if data_format == 'NCHW':
            image = tf.transpose(image, perm=(2, 0, 1))
        return image, labels, bboxes, bbox_img


def summaryImage(image, bboxes, name='image'):
    image = tf.expand_dims(image, 0)
    bboxes = tf.expand_dims(bboxes, 0)
    image_with_box = tf.image.draw_bounding_boxes(image, bboxes)
    tf.summary.image(name, image_with_box)


def randomSelect(x, func, num_cases):
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    for case in range(num_cases):
        return control_flow_ops.merge([
            func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
            for case in range(num_cases)])[0]
