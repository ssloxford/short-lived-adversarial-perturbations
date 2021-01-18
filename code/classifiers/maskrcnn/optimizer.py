import tensorflow as tf
from classifiers.Optimizer import Optimizer
import argparse


class MaskRCNNOptimizer(Optimizer):

    @staticmethod
    def get_input_tensor_name():
        return "image:0"

    def __init__(self, saver, sess, **kwargs):
        super(MaskRCNNOptimizer, self).__init__()
        args = argparse.Namespace(**kwargs)

        self.restore_path = args.restore_path
        self.class_dict = {}
        # self.new_size = args.new_size
        self.saver = saver
        self.sess = sess
        self.saver.restore(sess, self.restore_path)

        self.boxes = tf.compat.v1.get_default_graph().get_tensor_by_name("output/boxes:0")
        self._scores = tf.compat.v1.get_default_graph().get_tensor_by_name("output/scores:0")
        self.scores = tf.compat.v1.get_default_graph().get_tensor_by_name("fastrcnn_all_scores:0")
        self.labels = tf.compat.v1.get_default_graph().get_tensor_by_name("output/labels:0")
        self.input = tf.compat.v1.get_default_graph().get_tensor_by_name(self.get_input_tensor_name())

    @staticmethod
    def slapped_input_to_network_input(box_coords_tensor, slapped_input_tensor, **kwargs):
        # Maskrcnn expects BGR [0, 255] inputs, slapped_input_tensor is RGB
        tensor = slapped_input_tensor[0]
        tensor = tf.reverse(tensor, axis=[-1])
        return tensor*255

    def adjust_input_for_plot(self, input):
        return input.astype(float)/255.0


