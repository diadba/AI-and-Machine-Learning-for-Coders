"""
    Python Interpreter version: 3.11
    TensorFlow version: 2.14.0 (strictly below 2.15 for TensorFlow test guarantees)
    TensorFlow datasets version: 4.9.9
"""

import tensorflow as tf
import tensorflow_datasets as tfds

data, info = tfds.load("mnist", with_info=True)

filename="C:/users/acoots/tensorflow_datasets/mnist/3.0.1/mnist-test.tfrecord-00000-of-00001"
raw_dataset = tf.data.TFRecordDataset(filename)
for raw_record in raw_dataset:
    print(repr(raw_record))

# Create a description of the features.
feature_description = {
    "image": tf.io.FixedLenFeature([], dtype=tf.string),
    "label": tf.io.FixedLenFeature([], dtype=tf.int64),
}

def _parse_function(example_proto):
    # Parse the input 'tf.Example' proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_description)

parsed_dataset = raw_dataset.map(_parse_function)
for parsed_record in parsed_dataset.take(1):
    print((parsed_record))