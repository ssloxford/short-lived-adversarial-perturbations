import cv2


class Detector(object):
    """Object detector superclass, defining object detector interface.
        Detectors must subclass this in order to be usable with sentinet."""

    def __init__(self, *args, **kwargs):
        super(Detector, self).__init__()

    def forward(self, img_ori, tracker_box, tracker_pad, min_pad):
        raise NotImplementedError('Must be overridden')

    def preprocess_input(self, image):
        raise NotImplementedError('Must be overridden')

    def detect_and_draw_all(self, result, det_threshold, img, color=(0, 255, 0)):
        raise NotImplementedError('Must be overridden')

    def xrai(self, image, prediction_class, binarize=False, threshold=0.3):
        raise NotImplementedError('Must be implemented error')

    @staticmethod
    def needs_roi(self):
        raise NotImplementedError('Must be implemented error')

    def pad_tracker_box(self, img, tracker_box, tracker_pad, min_pad):

        (x, y, w, h) = [max(int(v), 0) for v in tracker_box]
        # add some padding
        pad_x = max(min_pad, int(tracker_pad * w))
        pad_y = max(min_pad, int(tracker_pad * h))
        x_pad = max(0, x - pad_x)
        y_pad = max(0, y - pad_y)
        w_pad = min(img.shape[1], w + 2 * pad_x)
        h_pad = min(img.shape[0], h + 2 * pad_y)

        tracker_roi = [x, y, x+w, y+h]
        tracker_roi_pad = [x_pad, y_pad, x_pad + w_pad, y_pad + h_pad]

        cutout = img[y_pad:y_pad + h_pad, x_pad:x_pad + w_pad]
        return cutout, tracker_roi, tracker_roi_pad

    @staticmethod
    def adjust_input_for_plot(input):
        return input


