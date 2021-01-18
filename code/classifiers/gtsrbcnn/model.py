import tensorflow as tf


class GtsrbCNNModel():

    def __new__(self, **kwargs):
        return self.build(**kwargs)

    @staticmethod
    def build(img_rows=32, img_cols=32, num_channels=3, n_classes=43, input_layer_name=None, custom_input=None):
        # define the model input
        if custom_input is not None:
            inputs = tf.keras.layers.Input(shape=(img_rows, img_cols, num_channels), tensor=custom_input)
        else:
            inputs = tf.keras.layers.Input(shape=(img_rows, img_cols, num_channels), name=input_layer_name)

        # 1st block
        x = tf.keras.layers.Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1),
                                   padding="same", input_shape=(img_rows, img_cols, num_channels))(inputs)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1),
                                   padding="same", input_shape=(img_rows, img_cols, num_channels))(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1),
                                   padding="same", input_shape=(img_rows, img_cols, num_channels))(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Dropout(.5)(x)
        conv1_flat = tf.keras.layers.Flatten()(x)

        # 2nd block
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1),
                           padding="same", input_shape=(img_rows, img_cols, num_channels))(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1),
                                   padding="same", input_shape=(img_rows, img_cols, num_channels))(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Dropout(.5)(x)
        conv2_flat = tf.keras.layers.Flatten()(x)

        # 3rd block
        x = tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1),
                                   padding="same", input_shape=(img_rows, img_cols, num_channels))(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1),
                                   padding="same", input_shape=(img_rows, img_cols, num_channels))(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)
        x = tf.keras.layers.Activation("relu", name="last_conv")(x)
        x = tf.keras.layers.Dropout(.5)(x)
        conv3_flat = tf.keras.layers.Flatten()(x)

        # fc 1
        x = tf.keras.layers.Concatenate(axis=1)([conv1_flat, conv2_flat, conv3_flat])
        x = tf.keras.layers.Dense(1024)(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Dropout(.5)(x)

        # fc 2
        x = tf.keras.layers.Dense(1024)(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Dropout(.5)(x)

        # softmax classifier
        x = tf.keras.layers.Dense(n_classes, name="last_fc")(x)
        x = tf.keras.layers.Activation("softmax", name="softmax")(x)

        model = tf.keras.models.Model(inputs, x, name="gtsrbcnn")

        return model