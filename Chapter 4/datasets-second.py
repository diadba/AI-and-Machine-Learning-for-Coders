"""
    Python Interpreter version: 3.11
    TensorFlow version: 2.15.0
    TensorFlow-addons version: 0.22.0
    TensorFlow datasets version: 4.9.9
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

print('tf.__version__:', tf.__version__)
print('tfds.__version__:', tfds.__version__)
print('tfa.__version__:', tfa.__version__)

data = tfds.load('horses_or_humans', split='train', as_supervised=True)
val_data = tfds.load('horses_or_humans', split='test', as_supervised=True)

def augmentimages(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 255)
    image = tf.image.random_flip_left_right(image)
    image = tfa.image.rotate(image, 40, interpolation='NEAREST')
    return image, label

train = data.map(augmentimages)
train_batches = train.shuffle(100).batch(32)
validation_branches = val_data.batch(32)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300,300,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_batches,
    epochs=10,
    validation_data=validation_branches,
    validation_steps=1
)