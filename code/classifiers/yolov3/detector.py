import tensorflow as tf
from classifiers import classifiers_utils as clfutils
import numpy as np
from classifiers.yolov3.yolov3_utils import restore_yolov3_session, parse_anchors, read_class_names
from classifiers.Detector import Detector
import cv2
import yaml


class Yolov3Model(Detector):

    def __init__(self, new_size=(416, 416), **kwargs):
        super(Yolov3Model, self).__init__()

        self.p = yaml.load(open("/home/code/classifiers/params.yaml", "r"), Loader=yaml.FullLoader)["yolov3"]["_base"]

        self.anchors = parse_anchors(self.p["yolov3_anchor_path"])
        self.classes = read_class_names(self.p["yolov3_class_name_path"])
        self.num_class = len(self.classes)

        self.class_dict = self.classes

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)
        self.new_size = new_size
        self.input_data, self.boxes, self.scores, self.labels, _, _ = restore_yolov3_session(
            self.sess, self.new_size, self.num_class, self.anchors, self.p["restore_path"],
            score_threshold=0.01)

    def needs_roi(self):
        return False

    def preprocess_input(self, image):
        height_ori, width_ori = image.shape[:2]
        img = cv2.resize(image, tuple(self.new_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.
        return img, height_ori, width_ori

    def forward(self, img_ori, tracker_box=None, tracker_pad=None, tracker_min_pad=None):
        (x, y, w, h) = [max(int(v), 0) for v in tracker_box]

        img, height_ori, width_ori = self.preprocess_input(img_ori)

        boxes_, scores_, labels_ = self.sess.run([self.boxes, self.scores, self.labels],
                                                 feed_dict={self.input_data: img})
        boxes_ = clfutils.scale_box_to_org_size(boxes_, width_ori, height_ori, tar_width=self.new_size[0], tar_height=self.new_size[1])

        tracker_roi = [x, y, x + w, y + h]

        return_dict = dict()
        return_dict["boxes"] = boxes_
        return_dict["scores"] = scores_
        return_dict["labels"] = labels_
        return_dict["tracker_roi"] = tracker_roi
        return_dict["input"] = img

        return return_dict

    def detect_and_draw_all(self, result, det_threshold, img, color=(0, 255, 0)):
        for j, box in enumerate(result["boxes"]):
            x0, y0, x1, y1 = box.flatten()
            score = result["scores"][j]
            label = result["labels"][j]
            class_name = self.class_dict[label]
            if score > det_threshold:
                print("Box at %s for %s (%s), score %s" % (box, class_name, label, score))
                clfutils.plot_one_box(img, [x0, y0, x1, y1], label="%s (%.3f)" % (class_name, score), color=color)
        return img
