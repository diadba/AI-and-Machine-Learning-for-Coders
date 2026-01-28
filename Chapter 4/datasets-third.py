"""
    Python Interpreter version: 3.11
    TensorFlow version: 2.14.0 (strictly below 2.15 for TensorFlow test guarantees)
    TensorFlow-addons version: 0.22.0
    TensorFlow datasets version: 4.9.9
"""

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.image_classification import cats_vs_dogs
import tensorflow_addons as tfa
import numpy as np

print('tf.__version__:', tf.__version__)
print('tfds.__version__:', tfds.__version__)
print('Default data directory:', tfds.core.constants.DATA_DIR)
print('tfa.__version__:', tfa.__version__)
print('np.__version__:', np.__version__)

# Monkeypatch as needed below.
_NAME_RE = cats_vs_dogs._NAME_RE
_NUM_CORRUPT_IMAGES = cats_vs_dogs._NUM_CORRUPT_IMAGES

def _patched_generate_examples(self, archive):
    num_skipped = 0
    for fname, fobj in archive:
        res = _NAME_RE.match(fname)
        if not res:
            continue
        label = res.group(1).lower()

        # Skip corrupt/non-JFIF images (TFDS expects a fixed count)
        if tf.compat.as_bytes("JFIF") not in fobj.peek(10):
            num_skipped += 1
            continue

        img_data = fobj.read()
        img_tensor = tf.image.decode_image(img_data)
        img_recoded = tf.io.encode_jpeg(img_tensor).numpy()

        record = {
            "image": img_recoded,
            "image/filename": fname,
            "label": label,
        }

        yield fname, record

    if num_skipped != _NUM_CORRUPT_IMAGES:
        raise ValueError(
            f"Expected {_NUM_CORRUPT_IMAGES} corrupt images, but found {num_skipped} instead."
        )

cats_vs_dogs.CatsVsDogs._generate_examples = _patched_generate_examples

def augmentimages(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 255)
    image = tf.image.resize(image, (300,300))
    return image, label

# setattr(tfds.image_classification.cats_vs_dogs, '_URL',"https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip")

# Early research shows a potential conflict with Python 3.11 and the subsequent four lines. Creating a monkeypatch.
count_data = tfds.load('cats_vs_dogs', split='train', as_supervised=True)
train_data = tfds.load('cats_vs_dogs', split='train[:80%]', as_supervised=True)
validation_data = tfds.load('cats_vs_dogs', split='train[80%:90%]', as_supervised=True)
test_data = tfds.load('cats_vs_dogs', split='train[-10%:]', as_supervised=True)

augmented_training_data = train_data.map(augmentimages)
train_batches = augmented_training_data.shuffle(1024).batch(32)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300,300,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(train_batches, epochs=25)