import tensorflow as tf
import numpy as np
from YOLOv3_TensorFlow.model import yolov3 as yolov3_model


def gpu_nms_batch(boxes, scores, num_classes, batch_size, max_boxes=50, score_thresh=0.5, nms_thresh=0.5):
    """
    Redefine gpu_nms for batch of images rather than single image.

    params:
        boxes: tensor of shape [batch_size, 10647, 4] # 10647=(13*13+26*26+52*52)*3, for input 416*416 image
        scores: tensor of shape [batch_size, 10647, num_classes], score=conf*prob
        num_classes: total number of classes
        max_boxes: integer, maximum number of predicted boxes you'd like, default is 50
        score_thresh: if [ highest class probability score < score_threshold]
                        then get rid of the corresponding box
        nms_thresh: real value, "intersection over union" threshold used for NMS filtering
    """

    boxes_list, label_list, score_list, batch_ind = [], [], [], []
    max_boxes = tf.constant(max_boxes, dtype='int32')

    # Step 1: Create a filtering mask based on "box_class_scores" by using "threshold".
    mask = tf.greater_equal(scores, tf.constant(score_thresh))
    for j in range(batch_size):

        batch_boxes = boxes[j]
        batch_scores = scores[j]
        batch_mask = mask[j]

        # Step 2: Do non_max_suppression for each class
        for i in range(num_classes):
            # Step 3: Apply the mask to scores, boxes and pick them out
            filter_boxes = tf.boolean_mask(batch_boxes, batch_mask[:, i])
            filter_score = tf.boolean_mask(batch_scores[:, i], batch_mask[:, i])
            nms_indices = tf.image.non_max_suppression(boxes=filter_boxes,
                                                       scores=filter_score,
                                                       max_output_size=max_boxes,
                                                       iou_threshold=nms_thresh, name='nms_indices')
            label_list.append(tf.ones_like(tf.gather(filter_score, nms_indices), 'int32') * i)
            boxes_list.append(tf.gather(filter_boxes, nms_indices))
            score_list.append(tf.gather(filter_score, nms_indices))
            batch_ind.append(tf.ones_like(tf.gather(filter_score, nms_indices), 'int32') * j)

    boxes = tf.concat(boxes_list, axis=0, name="concat_boxes")
    score = tf.concat(score_list, axis=0, name="concat_scores")
    label = tf.concat(label_list, axis=0, name="concat_labels")

    batch_ind = tf.concat(batch_ind, axis=0, name="concat_labels")

    return boxes, score, label, batch_ind


def restore_yolov3_session(sess, new_size, num_class, anchors, restore_path, batch_size=1, score_threshold=0.4):
    """
    creates yolov3 graph, restores session from checkpoint
    :return: symbolic references to input, boxes, scores and labels tensors.
    """
    input_data = tf.compat.v1.placeholder(tf.float32, [None, new_size[1], new_size[0], 3], name='input_data')
    yolo_model = yolov3_model(num_class, anchors)
    with tf.compat.v1.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(input_data)
    pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

    pred_scores = pred_confs * pred_probs

    boxes, scores, labels, binds = gpu_nms_batch(pred_boxes, pred_scores, num_class, batch_size, max_boxes=30,
                                                 score_thresh=score_threshold, nms_thresh=0.5)

    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, restore_path)

    return input_data, boxes, scores, labels, binds, saver


def parse_anchors(anchor_path):
    '''
    parse anchors.
    returned data: shape [N, 2], dtype float32
    '''
    anchors = np.reshape(np.asarray(open(anchor_path, 'r').read().split(','), np.float32), [-1, 2])
    return anchors


def read_class_names(class_name_path):
    names = {}
    with open(class_name_path, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names
