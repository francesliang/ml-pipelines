import os
import tensorflow as tf
import PIL
import numpy as np

NUM_CLASS = 10


def image_to_tfrecord(image_file, label_list):
    image = PIL.Image.open(image_file)
    image_arr = np.asarray(image)
    image_shape = image_arr.shape
    height, width = image_shape
    depth = 1

    result = None

    feature = {
        'height': _int64_feature([height]),
        'width': _int64_feature([width]),
        'depth': _int64_feature([depth]),
        'label': _int64_feature(label_list),
        'image_raw': _float_feature(image_arr.flatten())
    }
    result = tf.train.Example(features=tf.train.Features(feature=feature))

    return result


def write_tfrecords(tfrecord_file, image_labels):
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for file_name, label in list(image_labels.items()):
            label_list = np.zeros(NUM_CLASS, dtype=np.uint8)
            label_list[int(label)-1] = 1
            tf_example = image_to_tfrecord(file_name, label_list)
            if tf_example:
                writer.write(tf_example.SerializeToString())
            else:
                print("{} has mismatched shape".format(file_name))


def _bytes_feature(values):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

def _float_feature(values):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def _int64_feature(values):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))



