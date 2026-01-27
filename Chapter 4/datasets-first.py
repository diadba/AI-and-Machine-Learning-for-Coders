"""
    Python Interpreter version: 3.11
    TensorFlow version: 2.15.0
    TensorFlow datasets version: 4.9.9
"""

import tensorflow as tf
import tensorflow_datasets as tfds

print('tf.__version__:', tf.__version__)
print('tfds.__version__:', tfds.__version__)

(training_images, training_labels), (test_images, test_labels) = tfds.as_numpy(
    tfds.load(
        'fashion_mnist',
        split=['train', 'test'],
        batch_size=-1,
        as_supervised=True,
    )
)

training_images = training_images / 255.
test_images = test_images / 255.

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10,activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    training_images,
    training_labels,
    epochs=5
)