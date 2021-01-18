import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import os
import numpy as np
import skimage.transform


def remove_batch_dimension(features, label):
    return features[0], label[0]


def pad(x):
    max_pad = 2
    padd = (np.random.rand(4)*max_pad).astype(int)
    x = np.pad(x, ((padd[0], padd[1]), (padd[2], padd[3]), (0, 0)), mode="constant", constant_values=0)
    x = skimage.transform.resize(x, (32, 32), preserve_range=True)
    return x


def load_traffic_sign_ds(dir, n_classes, target_size=(32, 32), batch_size=None, augment=True):
    assert os.path.isdir(dir)
    if augment:
        img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            samplewise_center=True,
            shear_range=30,
            height_shift_range=5,
            width_shift_range=5,
            preprocessing_function=pad
        )
    else:
        img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=0,
            samplewise_center=True,
            shear_range=0,
            height_shift_range=0,
            width_shift_range=0,
        )
    gen = img_gen.flow_from_directory(dir, target_size, class_mode="categorical", batch_size=1)

    ds = tf.data.Dataset.from_generator(
        lambda: gen,
        output_types=(tf.float32, tf.float32),
        output_shapes=([1, target_size[0], target_size[1], 3], [1, n_classes])
    )
    ds = ds.map(remove_batch_dimension)
    if batch_size is not None:
        ds = ds.batch(batch_size, drop_remainder=True)

    return ds
