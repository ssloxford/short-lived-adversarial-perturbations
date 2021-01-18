import tensorflow as tf
import neural_structured_learning as nsl
from defences import loaders
import argparse
import yaml
import os
from classifiers.lisacnn.model import LisaCNNModel
from classifiers.gtsrbcnn.model import GtsrbCNNModel
from keras.backend.tensorflow_backend import set_session

tf.compat.v1.disable_eager_execution()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

DS_MAP = {
    "gtsrbcnn": "gtsrb",
    "lisacnn": "lisa",
}

MODEL_MAP = {
    "gtsrbcnn": GtsrbCNNModel,
    "lisacnn": LisaCNNModel,
}

N_CLASSES = {
    "gtsrbcnn": 43,
    "lisacnn": 18,
}


def convert_to_dictionaries(image, label):
    return {"image": image, "label": label}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-m", "--model_id", required=True, choices=["gtsrbcnn", "lisacnn"])
    parser.add_argument("-a", "--adversarial_training", action="store_true")
    parser.add_argument("-o", "--save_output_path", action="store", default="/home/code/defences/tmp/")
    parser.add_argument("--training_epochs", action="store", type=int)
    args = parser.parse_args()

    p = yaml.load(open("/home/code/defences/params.yaml", "r"), Loader=yaml.FullLoader)

    tp = argparse.Namespace(**p["training"])  # training parameters
    if args.training_epochs is not None:
        tp.epochs = args.training_epochs
    ds_name = DS_MAP[args.model_id]
    dp = argparse.Namespace(**p["dataset"][ds_name])  # dataset parameters

    train_ds = loaders.load_traffic_sign_ds("/home/data/datasets/%s/train/" % ds_name,
                                            n_classes=N_CLASSES[args.model_id], augment=True)
    test_ds = loaders.load_traffic_sign_ds("/home/data/datasets/%s/test/" % ds_name,
                                           n_classes=N_CLASSES[args.model_id], augment=False)
    val_ds = loaders.load_traffic_sign_ds("/home/data/datasets/%s/val/" % ds_name,
                                          n_classes=N_CLASSES[args.model_id], augment=False)

    train_ds = train_ds.batch(tp.batch_size)
    test_ds = test_ds.batch(tp.batch_size)
    val_ds = val_ds.batch(tp.batch_size)

    model = MODEL_MAP[args.model_id].build(input_layer_name="image")
    model.summary()

    adv = False
    if args.adversarial_training:
        ap = argparse.Namespace(**p["training"]["adv"])  # dataset parameters

        adv_config = nsl.configs.make_adv_reg_config(
            multiplier=ap.adv_multiplier,
            adv_step_size=ap.adv_step_size,
            adv_grad_norm=ap.adv_grad_norm)

        model = nsl.keras.AdversarialRegularization(
            model,
            label_keys=["label"],
            adv_config=adv_config)

        train_ds = train_ds.map(convert_to_dictionaries)
        val_ds = val_ds.map(convert_to_dictionaries)
        test_ds = test_ds.map(convert_to_dictionaries)
    optimizer = tf.keras.optimizers.Adam

    model.compile(optimizer=optimizer(learning_rate=tp.learning_rate),
                  loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    model.fit(train_ds, epochs=tp.epochs, validation_data=val_ds, validation_freq=tp.validation_freq,
              steps_per_epoch=int(dp.train_imgs / tp.batch_size), validation_steps=int(dp.val_imgs / tp.batch_size))

    result = model.evaluate(test_ds, steps=int(dp.test_imgs / tp.batch_size))

    os.makedirs(args.save_output_path, exist_ok=True)

    to_save = model
    model_name = args.model_id
    if args.adversarial_training:
        to_save = model.base_model  # AdversarialRegularization is just a wrapper
        model_name = model_name + "_adv"

    tf.keras.models.save_model(to_save, "%s/%s.h5" % (args.save_output_path, model_name), save_format="h5")

    print("saved output model at %s" % (args.save_output_path + model_name + ".h5"))






