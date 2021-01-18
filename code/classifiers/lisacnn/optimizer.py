import argparse
import pandas as pd
import tensorflow as tf
from classifiers.Optimizer import Optimizer
from classifiers import slap_graph_def as sgd


class LisaOptimizer(Optimizer):

    @staticmethod
    def get_input_tensor_name():
        return "Placeholder:0"

    def __init__(self, saver, sess, **kwargs):
        args = argparse.Namespace(**kwargs)
        self.restore_path = args.restore_path
        self.saver = saver
        self.sess = sess
        self.saver.restore(sess, self.restore_path)
        self.input = tf.compat.v1.get_default_graph().get_tensor_by_name(self.get_input_tensor_name())
        self.class_dict = {k: v for k, v in pd.read_csv(args.class_descr_fpath, index_col=False).values}
        self.learning_phase = tf.compat.v1.get_default_graph().get_tensor_by_name("keras_learning_phase:0")
        self.scores = tf.compat.v1.get_default_graph().get_tensor_by_name("Softmax_1:0")


    @staticmethod
    def slapped_input_to_network_input(box_coords_tensor, slapped_input_tensor, **kwargs):
        args = argparse.Namespace(**kwargs)
        lisa_input = []
        for j in range(args.batch_size):
            box = box_coords_tensor[j]
            img_with_ss = slapped_input_tensor[j]

            ss_box_cut = img_with_ss[box[1]:box[3], box[0]:box[2]]
            traffic_sign = tf.image.resize(ss_box_cut, (args.lisacnn_img_cols, args.lisacnn_img_rows),
                                                  method=sgd.interp_enum(args.interpolation_method, "tf"))
            traffic_sign = tf.clip_by_value(traffic_sign, 0, 1)  # inputs for lisacnn are [0, 1]
            lisa_input.append(tf.expand_dims(traffic_sign, 0))
        lisa_input = tf.concat(lisa_input, axis=0)
        return lisa_input

    def mandatory_feeds(self, **kwargs):
        return {self.learning_phase: False}
