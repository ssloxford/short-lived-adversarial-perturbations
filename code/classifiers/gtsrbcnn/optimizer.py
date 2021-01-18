import argparse
import pandas as pd
import tensorflow as tf
from classifiers.Optimizer import Optimizer
from classifiers import slap_graph_def as sgd

class GtsrbOptimizer(Optimizer):

    @staticmethod
    def get_input_tensor_name():
        return "features:0"

    def __init__(self, saver, sess, **kwargs):
        args = argparse.Namespace(**kwargs)
        self.restore_path = args.restore_path
        self.saver = saver
        self.sess = sess
        self.saver.restore(sess, self.restore_path)
        self.input = tf.compat.v1.get_default_graph().get_tensor_by_name(self.get_input_tensor_name())
        self.class_dict = {k: v for k, v in pd.read_csv(args.class_descr_fpath, index_col=False).values}
        self.placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("Placeholder:0")
        self.scores = tf.compat.v1.get_default_graph().get_tensor_by_name("Softmax:0")


    @staticmethod
    def slapped_input_to_network_input(box_coords_tensor, slapped_input_tensor, **kwargs):
        args = argparse.Namespace(**kwargs)
        gtsrb_input = []
        for j in range(args.batch_size):
            box = box_coords_tensor[j]
            img_with_ss = slapped_input_tensor[j]
            ss_box_cut = img_with_ss[box[1]:box[3], box[0]:box[2]]

            traffic_sign = tf.image.resize(ss_box_cut, (args.gtsrbcnn_img_cols, args.gtsrbcnn_img_rows),
                                                  method=sgd.interp_enum(args.interpolation_method, "tf"))
            traffic_sign = tf.clip_by_value(traffic_sign, 0, 1)
            traffic_sign = traffic_sign - 0.5
            gtsrb_input.append(tf.expand_dims(traffic_sign, 0))
        gtsrb_input = tf.concat(gtsrb_input, axis=0)
        return gtsrb_input

    def mandatory_feeds(self, **kwargs):
        return {self.placeholder: 1.0}

    def adjust_input_for_plot(self, input):
        return input+0.5
