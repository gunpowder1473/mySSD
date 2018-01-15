import tensorflow as tf
import os
from Tfrecord.make_tfrecord import GLOBAL_PATH

FILE_PATTERN = '%s_data.tfrecord'
CLASS_NUM = 10
ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'shape': 'Shape of the image',
    'bbox': 'A list of bounding boxes, one per each object.',
    'label': 'A list of labels, one per each object.',
}


slim = tf.contrib.slim

def getData(split_name='train', dataset_dir=GLOBAL_PATH, file_pattern=FILE_PATTERN,
            reader=None, items_to_descriptions=ITEMS_TO_DESCRIPTIONS, num_classes=CLASS_NUM):
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    if reader is None:
        reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'bbox/label': tf.VarLenFeature(dtype=tf.int64),
        'bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
        'bbox/truncated': tf.VarLenFeature(dtype=tf.int64),
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        # name_scope = bbox/+ymin ...
        'bbox': slim.tfexample_decoder.BoundingBox(['ymin', 'xmin', 'ymax', 'xmax'], 'bbox/'),
        'label': slim.tfexample_decoder.Tensor('bbox/label'),
        'difficult': slim.tfexample_decoder.Tensor('bbox/difficult'),
        'truncated': slim.tfexample_decoder.Tensor('bbox/truncated'),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    # slim中Dataset生成一个字典，参数定义如下：
    # Args:
    #   data_sources: A list of files that make up the dataset.
    #   reader: The reader class, a subclass of BaseReader such as TextLineReader
    #     or TFRecordReader.
    #   decoder: An instance of a data_decoder.
    #   num_samples: The number of samples in the dataset.
    #   items_to_descriptions: A map from the items that the dataset provides to
    #     the descriptions of those items.
    #   **kwargs: Any remaining dataset-specific fields.
    return slim.dataset.Dataset(data_sources=file_pattern, reader=reader, decoder=decoder,
                                num_samples=25600, items_to_descriptions=items_to_descriptions,
                                num_classes=num_classes, labels_to_names=None)


