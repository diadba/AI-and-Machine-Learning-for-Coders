"""
    Python Interpreter version: 3.11
    TensorFlow version: 2.14.0 (strictly below 2.15 for TensorFlow test guarantees)
    TensorFlow-addons version: 0.22.0
    TensorFlow datasets version: 4.9.9
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import multiprocessing

# Creating the model architecture
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300,300,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
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

    return model

### Extract (ETL)
train_data = tfds.load('cats_vs_dogs', split='train', with_info=True)

# Sharding.
file_pattern = f'C:/users/{user}/tensorflow_datasets/cats_vs_dogs/4.0.1/cats_vs_dogs-train.tfrecord*'
files = tf.data.Dataset.list_files(file_pattern)

train_dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=4, num_parallel_calls=tf.data.experimental.AUTOTUNE)

### Transform (ETL)
def read_tfrecord(serialized_example):
    feature_description = {
        "image": tf.io.FixedLenFeature((), tf.string, ""),
        "label": tf.io.FixedLenFeature((), tf.int64, -1),
    }

    example = tf.io.parse_single_example(serialized_example, feature_description)
    image = tf.io.decode_jpeg(example['image'], channels=3)
    image = tf.cast(image, tf.float32)
    image = image / 255
    image = tf.image.resize(image, (300,300))
    return image, example['label']

cores = multiprocessing.cpu_count()
print(cores)

train_dataset = train_dataset.map(read_tfrecord, num_parallel_calls=cores)

### Load (ETL)
train_dataset = train_dataset.shuffle(1024).batch(32)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

model = create_model()
model.fit(
    train_dataset,
    epochs=10,
    verbose=1
)