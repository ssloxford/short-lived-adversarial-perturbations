import tensorflow as tf
from YOLOv3_TensorFlow.model import yolov3 as yolov3_model
from classifiers.Optimizer import Optimizer
from classifiers.yolov3.yolov3_utils import parse_anchors, read_class_names, gpu_nms_batch
import argparse
import numpy as np


def add_yolov3_head_instructions(batch_size, new_size, num_class, anchors):
    assert batch_size>=1
    fm1 = tf.compat.v1.get_default_graph().get_tensor_by_name("yolov3/yolov3_head/feature_map_1:0")
    fm2 = tf.compat.v1.get_default_graph().get_tensor_by_name("yolov3/yolov3_head/feature_map_2:0")
    fm3 = tf.compat.v1.get_default_graph().get_tensor_by_name("yolov3/yolov3_head/feature_map_3:0")

    # initialize some required parameters
    yolo_model = yolov3_model(num_class, anchors)
    yolo_model.img_size = np.array(new_size)

    # connect the meta-graph imported with the second part of the yolo network
    pred_boxes, pred_confs, pred_probs = yolo_model.predict([fm1, fm2, fm3])
    pred_scores = pred_confs * pred_probs

    return pred_boxes, pred_confs, pred_probs, pred_scores


class Yolov3Optimizer(Optimizer):

    @staticmethod
    def get_input_tensor_name():
        return "input_data:0"

    def __init__(self, saver, sess, **kwargs):
        args = argparse.Namespace(**kwargs)

        self.anchors = parse_anchors(args.yolov3_anchor_path)
        self.classes = read_class_names(args.yolov3_class_name_path)
        self.num_class = len(self.classes)
        self.restore_path = args.restore_path
        self.class_dict = {}
        self.new_size = args.yolov3_new_size
        self.saver = saver
        self.sess = sess
        self.saver.restore(sess, self.restore_path)
        self.input = tf.compat.v1.get_default_graph().get_tensor_by_name(self.get_input_tensor_name())

        self.boxes, self.confs, self.probs, self.scores = add_yolov3_head_instructions(
            args.batch_size, self.new_size, self.num_class, self.anchors)

    @staticmethod
    def slapped_input_to_network_input(box_coords_tensor, slapped_input_tensor, **kwargs):
        return slapped_input_tensor

