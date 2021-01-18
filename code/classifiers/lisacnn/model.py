import tensorflow as tf


class LisaCNNModel():

    def __new__(self, **kwargs):
        return self.build(**kwargs)

    @staticmethod
    def build(img_rows=32, img_cols=32, num_channels=3, n_classes=18, nb_filters=64, input_layer_name=None, custom_input=None):

        # define the model input
        if custom_input is not None:
            inputs = tf.keras.layers.Input(shape=(img_rows, img_cols, num_channels), tensor=custom_input)
        else:
            inputs = tf.keras.layers.Input(shape=(img_rows, img_cols, num_channels), name=input_layer_name)

        x = tf.keras.layers.Conv2D(filters=nb_filters, kernel_size=(8, 8), strides=(2, 2),
                            padding="same", input_shape=(img_rows, img_cols, num_channels))(inputs)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(filters=nb_filters * 2, kernel_size=(6, 6), strides=(2, 2), padding="valid")(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(filters=nb_filters * 2, kernel_size=(5, 5), strides=(1, 1), padding="valid")(x)
        x = tf.keras.layers.Activation("relu", name="last_conv")(x)
        x = tf.keras.layers.Flatten()(x)

        # softmax classifier
        x = tf.keras.layers.Dense(n_classes, name="last_fc")(x)
        x = tf.keras.layers.Activation("softmax", name="softmax")(x)

        model = tf.keras.models.Model(inputs, x, name="lisacnn")
        return model