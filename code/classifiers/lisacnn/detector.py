import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from classifiers import classifiers_utils as clfutils
from classifiers.Detector import Detector
import saliency


def preprocess_input_cvpr18(image):
    assert image.dtype==np.uint8
    assert image.ndim==3
    return image.astype(np.float32)/255.0


def adjust_input_for_plot_cvpr18(image):
    return (image*255).astype(np.uint8)


def adjust_input_for_plot_def(image):
    image = image + image.min()
    image = image / image.max()
    image = image * 255
    image = image.astype(np.uint8)
    return np.clip(image, 0, 255)


def preprocess_input_def(image):
    assert image.dtype==np.uint8
    assert image.ndim==3
    img = image.astype(np.float32)/255.0  # apply re-scaling of ImageDataGenerator
    img = img - np.mean(img)  # apply samplewise_center of ImageDataGenerator
    return img


PARAMS = {
    "cvpr18": {
        "meta_graph_path": "/home/data/models/lisacnn/cvpr18/all_r_ivan.ckpt.meta",
        "restore_path": "/home/data/models/lisacnn/cvpr18/all_r_ivan.ckpt",
        "class_descr_path": "/home/data/models/lisacnn/cvpr18/classes_to_sign_descr.csv",
        "input_tensor_name": "Placeholder:0",
        "output_tensor_name": "Softmax_1:0",
        "last_conv_name": "Relu_5:0",
        "last_fc_name": "add_7:0",
        "add_feed_tensors": {
            "keras_learning_phase:0": False
        },
        "preprocess_input": preprocess_input_cvpr18,
        "adjust_input_for_plot": adjust_input_for_plot_cvpr18,
    },
    "cvpr18iyswim": {
        "meta_graph_path": "/home/data/models/lisacnn/cvpr18iyswim/model.meta",
        "restore_path": "/home/data/models/lisacnn/cvpr18iyswim/model",
        "class_descr_path": "/home/data/models/lisacnn/cvpr18/classes_to_sign_descr.csv",
        "input_tensor_name": "input:0",
    },
    "usenix21": {
        "meta_graph_path": "/home/data/models/lisacnn/usenix21/lisacnn_scratch.meta",
        "restore_path": "/home/data/models/lisacnn/usenix21/lisacnn_scratch",
        "class_descr_path": "/home/data/models/lisacnn/usenix21/classes_to_sign_descr_slap.csv",
        "input_tensor_name": "image:0",
        "output_tensor_name": "softmax/Softmax:0",
        "last_conv_name": "last_conv/Relu:0",
        "last_fc_name": "last_fc/BiasAdd:0",
        "add_feed_tensors": {},
        "preprocess_input": preprocess_input_def,
        "adjust_input_for_plot": adjust_input_for_plot_def,
    },
    "usenix21adv": {
        "meta_graph_path": "/home/data/models/lisacnn/usenix21adv/lisacnn_adv.meta",
        "restore_path": "/home/data/models/lisacnn/usenix21adv/lisacnn_adv",
        "class_descr_path": "/home/data/models/lisacnn/usenix21adv/classes_to_sign_descr_slap.csv",
        "input_tensor_name": "image:0",
        "output_tensor_name": "softmax/Softmax:0",
        "last_conv_name": "last_conv/Relu:0",
        "last_fc_name": "last_fc/BiasAdd:0",
        "add_feed_tensors": {},
        "preprocess_input": preprocess_input_def,
        "adjust_input_for_plot": adjust_input_for_plot_def,
    }
}


class LisaCNNModel(Detector):

    def __init__(self, saliency=False, model_id1=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # load base parameters
        self.p = PARAMS["cvpr18"]
        if model_id1 is not None:
            assert model_id1 in PARAMS, "model_id1 %s is not in parameters in lisacnn/detector.py" % model_id1
            self.p.update(PARAMS[model_id1])
        p = self.p

        # output labels
        self.class_dict = {k: v for k, v in pd.read_csv(p["class_descr_path"], index_col=False).values}
        
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        # setup session and graph
        self.sess = tf.compat.v1.Session(config=config)
        self.saver = tf.compat.v1.train.import_meta_graph(p["meta_graph_path"])
        self.saver.restore(self.sess, p["restore_path"])

        # link tensors
        self.input = tf.compat.v1.get_default_graph().get_tensor_by_name(p["input_tensor_name"])
        self.output = tf.compat.v1.get_default_graph().get_tensor_by_name(p["output_tensor_name"])
        self.last_conv = tf.compat.v1.get_default_graph().get_tensor_by_name(p["last_conv_name"])
        self.last_fc = tf.compat.v1.get_default_graph().get_tensor_by_name(p["last_fc_name"])
        self.feed_tensors = p["add_feed_tensors"]
        self.prob = self.output
        self.preprocess_input = p["preprocess_input"]
        self.adjust_input_for_plot = p["adjust_input_for_plot"]

        # saliency setup
        self.saliency = False
        if saliency:
            self.saliency = True
            self.neuron_selector = tf.compat.v1.placeholder(tf.int32)
            self.saliency_target = self.last_fc[0][self.neuron_selector]
            self.prediction = tf.argmax(self.last_fc, 1)

    @staticmethod
    def load_image(path):
        img = cv2.imread(path)[..., [2, 1, 0]]
        img = cv2.resize(img, (32, 32), cv2.INTER_LINEAR)
        return img

    def needs_roi(self):
        return True

    def xrai(self, image, prediction_class, binarize=False, threshold=0.3):

        image = self.preprocess_input(image)
        if not hasattr(self, 'xrai_object'):
            self.xrai_object = saliency.XRAI(tf.compat.v1.get_default_graph(), self.sess, self.saliency_target, self.input)

        feed_dict = {**dict({self.neuron_selector: prediction_class}), **self.feed_tensors}
        xrai_attributions = self.xrai_object.GetMask(image, feed_dict=feed_dict)

        # most salient 30%
        xrai_salient_mask = xrai_attributions > np.percentile(xrai_attributions, (1-threshold)*100)
        xrai_im_mask = np.ones(image.shape)
        xrai_im_mask[~xrai_salient_mask] = 0

        if binarize:
            xrai_im_mask = (xrai_im_mask > 0).astype(bool)
        return xrai_im_mask

    def forward(self, img_ori, tracker_box=None, tracker_pad=None, tracker_min_pad=None, save_image=True, probs_only=False, prediction_only=False):
        cutout = np.copy(img_ori)
        tracker_roi = None
        tracker_roi_pad = None
        if tracker_box is not None and tracker_pad is not None and tracker_min_pad is not None:
            cutout, tracker_roi, tracker_roi_pad = self.pad_tracker_box(img_ori, tracker_box, tracker_pad, tracker_min_pad)

        cutout_r = cv2.resize(cutout, (32, 32), cv2.INTER_LINEAR)
        cutout_r_preproc = self.preprocess_input(cutout_r)
        feed_dict = {**dict({self.input: cutout_r_preproc[np.newaxis, ...]}), **self.feed_tensors}
        labels_out_ = self.sess.run(self.output, feed_dict=feed_dict)

        if prediction_only:
            prediction = np.argmax(labels_out_[0])
            return prediction, labels_out_[0][prediction]

        if probs_only:
            return np.array(labels_out_[0])

        return_dict = dict()
        return_dict["labels"] = labels_out_.flatten()
        return_dict["tracker_roi"] = tracker_roi
        return_dict["tracker_roi_pad"] = tracker_roi_pad
        return_dict["input"] = cutout_r_preproc

        return return_dict

    def detect_and_draw_all(self, result, det_threshold, img, color=(0, 255, 0)):
        class_name = self.class_dict[np.argmax(result["labels"])]
        score = result["labels"].max()
        label = np.argmax(result["labels"])
        print("Box at %s for %s (%s), score %s" % (result["tracker_roi"], class_name, label, score))
        clfutils.plot_one_box(img, result["tracker_roi"], label="%s (%.3f)" % (class_name, score), color=color)
        # add tracked roi to top right of image
        img[:32:, img.shape[1] - 32:] = self.adjust_input_for_plot(result["input"])
        return img
