import tensorflow as tf
from classifiers import classifiers_utils as clfutils
import numpy as np
import cv2
from classifiers.Detector import Detector
import yaml


class MaskRCNN(Detector):

    def __init__(self, new_size=(416, 416), **kwargs):
        super(MaskRCNN, self).__init__()
        self.p = yaml.load(open("/home/code/classifiers/params.yaml", "r"), Loader=yaml.FullLoader)["maskrcnn"]["_base"]
        
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)
        self.class_dict = {i: c for i, c in enumerate(['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                                                      'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
                                                      'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                                                      'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
                                                      'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                                                      'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                                                      'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                                                      'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                                                      'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                                                      'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                                                      'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                                                      'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                                                      'hair drier', 'toothbrush'])}

        saver = tf.compat.v1.train.import_meta_graph(self.p["meta_graph_path"])
        saver.restore(self.sess, self.p["restore_path"])

        self.boxes = tf.compat.v1.get_default_graph().get_tensor_by_name("output/boxes:0")
        self.scores = tf.compat.v1.get_default_graph().get_tensor_by_name("output/scores:0")
        self._scores = tf.compat.v1.get_default_graph().get_tensor_by_name("fastrcnn_all_scores:0")
        self.labels = tf.compat.v1.get_default_graph().get_tensor_by_name("output/labels:0")
        self.input = tf.compat.v1.get_default_graph().get_tensor_by_name("image:0")

        self.new_size = new_size

    def needs_roi(self):
        return False

    def preprocess_input(self, image):
        assert image.dtype == np.uint8, image.dtype
        assert image.ndim == 3, image.ndim
        height_ori, width_ori = image.shape[:2]
        img = cv2.resize(image, (self.new_size[0], self.new_size[1]), cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img, height_ori, width_ori

    def forward(self, img_ori, tracker_box=None, tracker_pad=None, tracker_min_pad=None):

        (x, y, w, h) = [max(int(v), 0) for v in tracker_box]

        img, height_ori, width_ori = self.preprocess_input(img_ori)

        boxes_, scores_, labels_, _scores_ = self.sess.run(
            [self.boxes, self.scores, self.labels, self._scores],
            feed_dict={self.input: img})

        boxes_ = clfutils.scale_box_to_org_size(boxes_, width_ori, height_ori, tar_width=self.new_size[0], tar_height=self.new_size[1])

        tracker_roi = [x, y, x + w, y + h]

        return_dict = dict()
        return_dict["boxes"] = boxes_
        return_dict["scores"] = scores_
        return_dict["_scores"] = _scores_
        return_dict["labels"] = labels_
        return_dict["tracker_roi"] = tracker_roi
        return_dict["input"] = img

        return return_dict

    @staticmethod
    def adjust_input_for_plot(input):
        return input[..., [2, 1, 0]].astype(float)/255.0

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
