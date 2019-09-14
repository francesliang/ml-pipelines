import os
import tensorflow as tf


def image_to_tfrecord(image_bytes, label):
    image_shape = tf.image.decode_png(image_bytes).shape

    feature = {
        #'height': _int64_feature(image_shape[0]),
        #'width': _int64_feature(image_shape[1]),
        #'depth': _int64_feature(image_shape[2]),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_bytes)
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecords(tfrecord_file, image_labels):
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for file_name, label in image_labels.items():
            image_bytes = open(file_name, 'rb').read()
            tf_example = image_to_tfrecord(image_bytes, label)
            writer.write(tf_example.SerializeToString())


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



