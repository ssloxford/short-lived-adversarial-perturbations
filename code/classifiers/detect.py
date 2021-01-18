import argparse
import os
import cv2
from classifiers.gtsrbcnn.detector import GtsrbCNNModel
from classifiers.lisacnn.detector import LisaCNNModel
from classifiers.yolov3.detector import Yolov3Model
from classifiers.maskrcnn.detector import MaskRCNN
import skimage.io


MODEL_MAP = {
    "yolov3": Yolov3Model,
    "gtsrbcnn": GtsrbCNNModel,
    "lisacnn": LisaCNNModel,
    "maskrcnn": MaskRCNN
}

_SUPPORTED_FMTS = [".jpeg", ".ppm", ".jpg", ".png"]

DETECTION_THRESHOLD = {
    "yolov3": 0.4,
    "gtsrbcnn": 0.5,
    "lisacnn": 0.5,
    "maskrcnn": 0.6
}


def check_file_format(filepath):
    """
    Checks whether filepath is in the supported formats, also returns whether filepath is an image
    """
    filename, file_extension = os.path.splitext(filepath)
    file_extension = file_extension.lower()
    assert file_extension in _SUPPORTED_FMTS, "unsupported format for %s, must be one of %s" % (filepath, _SUPPORTED_FMTS)
    return True


def test_X_available():
    import matplotlib
    matplotlib.use("tkagg")
    import matplotlib.pyplot as plt
    try:
        plt.plot([],[])
        return True
    except Exception as e:
        return False


def check_roi_valid(shape, roi):
    """
    :param shape: (h, w)
    :param roi: (x, y, roi_w, roi_h)
    :return:
    """
    assert 0 <= roi[0] <= shape[1], "roi of %s malformed for image of shape %s" % (roi, shape)
    assert 0 <= roi[1] <= shape[0], "roi of %s malformed for image of shape %s" % (roi, shape)
    assert 0 <= roi[0] + roi[2] <= shape[1], "roi of %s malformed for image of shape %s" % (roi, shape)
    assert 0 <= roi[1] + roi[3] <= shape[0], "roi of %s malformed for image of shape %s" % (roi, shape)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference for images or videos.")
    parser.add_argument("-f", "--filepath", type=str, required=True, help="Run for specific file")
    parser.add_argument("-n0", "--detector_name", type=str, required=True,
                        choices=["lisacnn", "gtsrbcnn", "yolov3", "maskrcnn"])
    parser.add_argument("-n1", "--detector_id1", type=str, default=None,
                        choices=["cvpr18", "cvpr18iyswim", "usenix21", "usenix21adv"])
    parser.add_argument("-o", "--output_folder", action="store", default="/home/code/classifiers/tmp/",
                        help="Save output to this path")
    parser.add_argument('--roi', nargs=4, type=int, help='Roi coordinates in (x, y, w, h)')
    parser.add_argument('--roi_pad', action="store", type=float, default=0.1)
    parser.add_argument('--roi_min_pad', action="store", type=int, default=4)

    args = parser.parse_args()

    args.filepath = os.path.abspath(args.filepath)
    args.filename = args.filepath.split(os.path.sep)[-1]
    check_file_format(args.filepath)
    full_detector_name = "".join([args.detector_name, ("_" + args.detector_id1 if args.detector_id1 else "")])

    # load model based on the detector name
    model = MODEL_MAP[args.detector_name](model_id1=args.detector_id1)
    args.detection_threshold = DETECTION_THRESHOLD[args.detector_name]

    img = cv2.imread(args.filepath)[..., [2, 1, 0]]  # opencv reads BGR, models are fed RGB

    if model.needs_roi():
        if args.roi is None:
            is_display_available = test_X_available()
            if not is_display_available:
                print("Display is not available. Either provide a roi to this script with '--roi x y w h' or make"
                      "sure display is available within the container")
                exit()
            else:
                # open an opencv window asking the user to draw a 2D-box around the target object
                cv2.namedWindow("select_roi")
                cv2.moveWindow("select_roi", 0, 0)
                roi = cv2.selectROI("select_roi", img[..., [2, 1, 0]], fromCenter=False, showCrosshair=True)
                cv2.destroyWindow("select_roi")
    else:
        args.roi = [0, 0, 1, 1]
    check_roi_valid(img.shape[:2], args.roi)

    # run inference
    rdict = model.forward(img_ori=img, tracker_box=args.roi, tracker_pad=args.roi_pad, tracker_min_pad=args.roi_min_pad)
    # copy image to draw boxes on it
    img1 = img.copy()
    # draw all boxes
    model.detect_and_draw_all(rdict, .01, img1)

    # save output
    os.makedirs(args.output_folder, exist_ok=True)
    output_filename = os.path.join(args.output_folder, args.filename)
    skimage.io.imsave(output_filename, img1)

    cv2.destroyAllWindows()
