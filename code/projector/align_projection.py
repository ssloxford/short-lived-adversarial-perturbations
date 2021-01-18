#!/usr/bin/python3
# coding: utf-8

from __future__ import division, print_function

import argparse
import json
import os
import shutil
import time

import sys
import cv2
import numpy as np
import subprocess
from projector import projector_utils as putils
import matplotlib.pyplot as plt
from slap_utils import sutils
import imutils
from object_detectors.gtsrbcnn.detector import GtsrbCNNModel
from object_detectors.lisacnn.detector import LisaCNNModel
from object_detectors.yolov3.detector import Yolov3Model
from object_detectors.maskrcnn.detector import MaskRCNN
from object_detectors.Detector import Detector


PROJ_WHITE_PAD = 10


def overlay_at(base_img, overlay_img, top_left_corner):
    oh, ow = overlay_img.shape[:2]

    bot_right_corner_x = top_left_corner[0] + ow
    bot_right_corner_y = top_left_corner[1] + oh

    assert bot_right_corner_x < base_img.shape[1], "%d, %d" % (bot_right_corner_x, base_img.shape[1])
    assert bot_right_corner_y < base_img.shape[0], "%d, %d" % (bot_right_corner_y, base_img.shape[0])

    base_img_copy = base_img.copy()
    base_img_copy[top_left_corner[1]:bot_right_corner_y, top_left_corner[0]:bot_right_corner_x, :] = overlay_img
    return base_img_copy


def rotateImage(image, angle, image_center=None, interp=cv2.INTER_CUBIC):
    if image_center is None:
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=interp)
    return result


def get_w_content(pbase, pwh, pww, piw, pih, pix, piy, pnts, ra, bo, interp, ch_offsets, pad_color):
    window_content = np.zeros((pwh, pww, 3), dtype=np.uint8)
    # proj_pad = sutils.persp_transform_pad(pbase, pad=.1, color=(255, 255, 255))

    mask = np.sum(pbase, axis=-1)

    mask[mask!=0] = 1.0
    pbase_adjusted = pbase.copy().astype(float)

    for i, _ in enumerate(ch_offsets):
        pbase_adjusted[..., i] = pbase[..., i] + ch_offsets[i]

    pbase_adjusted = pbase_adjusted * mask[..., np.newaxis]

    pbase_adjusted = np.clip(pbase_adjusted, 0, 255).astype(np.uint8)

    pbase_YCrCb = cv2.cvtColor(pbase_adjusted, cv2.COLOR_BGR2YCrCb)
    pbase_YCrCb[..., 0] = np.clip(pbase_YCrCb[..., 0] + int(bo), 0, 255).astype(np.uint8)
    pbase_rgb = cv2.cvtColor(pbase_YCrCb, cv2.COLOR_YCrCb2BGR)

    proj_rsz = cv2.resize(pbase_rgb, (piw, pih), interp)
    proj_rsz_pad = sutils.persp_transform_pad(proj_rsz, pad=PROJ_WHITE_PAD, color=pad_color)

    window_content = overlay_at(window_content, proj_rsz_pad, (pix, piy))
    window_content = rotateImage(window_content, ra, (pix+(piw/2), piy+(pih/2)), interp=interp)

    window_content = sutils.four_point_transform(window_content, pnts)
    window_content = cv2.resize(window_content, (pww, pwh), interp)
    return window_content, pbase_adjusted


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Help (manually) align the projection image onto the stop sign\n"
                                                 "python align_projection.py -id <ID> -n <OPT_NAME> -pm <LUX>")
    parser.add_argument("-id", "--experiment_id", type=str, required=True)
    parser.add_argument("-n", "--detector_name", action="store", required=True, type=str, help="Optimizer name")
    parser.add_argument("-pm", "--proj_model_lux", action="store", required=True, type=str, help="Projection model lux id")
    parser.add_argument("-r", "--force_redo", action="store_true")
    args = parser.parse_args()

    this_script_name = os.path.basename(__file__).split(".")[0]
    oname = args.detector_name
    pm_lux = args.proj_model_lux
    experiment_folder = os.path.join("/home/data/%s" % args.experiment_id)

    params = vars(args)

    # first thing check exposure value
    cut_ss_params = json.load(open(os.path.join(experiment_folder, "cut_stop_sign", args.proj_model_lux, "params.json"), "r"))
    putils.set_exposure_auto(1)
    exposure_v = putils.get_exposure_abs()
    if "exposure" in cut_ss_params:
        exposure_v = putils.set_exposure_abs(cut_ss_params["exposure"])
    webcam = cv2.VideoCapture(0)

    alignment_folder = os.path.join(experiment_folder, "alignment")
    os.makedirs(alignment_folder, exist_ok=True)

    main_screen = (1920, 1080)
    proj_screen = (1024, 768)

    pw_w = proj_screen[0]
    pw_h = proj_screen[1] - 100
    pi_w = 256
    pi_h = 256
    pi_x = pw_w // 2-pi_w
    pi_y = pw_h // 2-pi_h
    rot_angle = 0
    pts = np.array([[0, 0], [pw_w, 0], [0, pw_h], [pw_w, pw_h]], dtype=int)
    brightness_offset = 0

    if os.path.isfile(os.path.join(alignment_folder, "params.json")) and not args.force_redo:
        try:
            prev_params = json.load(open(os.path.join(alignment_folder, "params.json"), "r"))
            pi_w = prev_params["projection_window_atk"]["pi_w"]
            pi_h = prev_params["projection_window_atk"]["pi_h"]
            pi_x = prev_params["projection_window_atk"]["pi_x"]
            pi_y = prev_params["projection_window_atk"]["pi_y"]
            rot_angle = prev_params["projection_window_atk"]["rot_angle"]
            pts = np.array(prev_params["projection_window_atk"]["pts"]).reshape(4, 2)
            brightness_offset = int(prev_params["projection_window_atk"]["brightness_offset"])
            interp = int(prev_params["projection_window_atk"]["interpolation"])
        except KeyError as e:
            pass

    optimizer_folder = os.path.join(experiment_folder, "optimize", pm_lux, oname)
    opt_params = json.load(open(os.path.join(optimizer_folder, "params.json"), "r"))
    proj_rgb_f = np.load(os.path.join(optimizer_folder, "projections.npy"))[-1]
    proj_base = (proj_rgb_f * 255)[..., ::-1].astype(np.uint8)

    cv2.namedWindow("camera-input")
    cv2.moveWindow("camera-input", 20, 20)

    interpolations = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST, cv2.INTER_AREA]
    i_names = ["linear", "cubic", "nearest", "area"]

    model = None
    detection_on = False
    hide = False
    interpolation = 0
    ch_offsets = [0, 0, 0]
    network_input = False
    pad_color = (255, 255, 255)

    try:

        while True:
            ret, frame = webcam.read()

            window_content, _adj = get_w_content(proj_base, pw_h, pw_w, pi_w, pi_h, pi_x, pi_y, pts, rot_angle,
                                           brightness_offset, interpolations[interpolation], ch_offsets, pad_color)

            key = cv2.waitKey(1)

            if key == ord('c') or key == 13:
                cv2.destroyAllWindows()
                break
            if key == ord('w'):
                pi_y = max(pi_y-5, 0)
            elif key == ord('s'):
                pi_y = min(pi_y+5, pw_h-pi_h - PROJ_WHITE_PAD * 2-1)
            elif key == ord('a'):
                pi_x = max(pi_x-5, 0)
            elif key == ord('d'):
                pi_x = min(pi_x + 5, pw_w - pi_w - PROJ_WHITE_PAD *2 -1)
            elif key == ord('q'):
                rot_angle = (rot_angle - 1) % 360
            elif key == ord('e'):
                rot_angle = (rot_angle + 1) % 360
            elif key == ord('x'):
                pi_w = min(pi_w + 3, pw_w - pi_x - PROJ_WHITE_PAD * 2 - 1)
            elif key == ord('z'):
                pi_w = max(pi_w-3, 10)
            elif key == ord('y'):
                pi_h = min(pi_h+3, pw_h-pi_y-PROJ_WHITE_PAD*2-1)
            elif key == ord('t'):
                pi_h = max(pi_h-3, 10)
            elif key == ord('h'):
                hide = not hide
            elif key == ord('2'):
                brightness_offset = min(50, brightness_offset+1)
            elif key == ord('1'):
                brightness_offset = max(-100, brightness_offset-1)
            elif key == ord('3'):
                detection_on = True
                if model is None or type(model).__name__ != "Yolov3Model":
                    sys.stdout.write("Loading Yolov3 model, be patient\n")
                    model = Yolov3Model()
                    sys.stdout.write("Done\n")
            elif key == ord('4'):
                detection_on = True
                if model is None or type(model).__name__ != "MaskRCNN":
                    sys.stdout.write("Loading MaskRCNN model, be patient\n")
                    model = MaskRCNN()
                    sys.stdout.write("Done\n")
            elif key == ord('p'):
                detection_on = False
            elif key == ord('f'):
                pts = np.array([[0, 0], [pw_w, 0], [0, pw_h], [pw_w, pw_h]], dtype=int)
                window_content, adj_ = get_w_content(proj_base, pw_h, pw_w, pi_w, pi_h, pi_x, pi_y, pts, rot_angle,
                                               brightness_offset, interpolations[interpolation], ch_offsets, pad_color)
                cv2.imshow("projection", window_content)
                _, pts = sutils.fix_perspective_mouse2(window_content, "projection")
                cv2.imshow("projection", window_content)
            elif key == ord('7'):
                exposure_v = putils.set_exposure_abs(max(1, exposure_v-5))
            elif key == ord('8'):
                exposure_v = putils.set_exposure_abs(min(2047, exposure_v+5))
            elif key == ord('9'):
                interpolation = (interpolation+1)% len(interpolations)
            elif key == ord('0'):
                proj_rgb_f = np.load(os.path.join(optimizer_folder, "projections.npy"))[-1]
                proj_base = (proj_rgb_f * 255)[..., ::-1].astype(np.uint8)
            elif key == ord('u'):
                ch_offsets[0] = max(ch_offsets[0]-2, -150)
            elif key == ord('i'):
                ch_offsets[0] = min(ch_offsets[0]+2, 150)
            elif key == ord('j'):
                ch_offsets[1] = max(ch_offsets[1] - 2, -150)
            elif key == ord('k'):
                ch_offsets[1] = min(ch_offsets[1] + 2, 150)
            elif key == ord('n'):
                ch_offsets[2] = max(ch_offsets[2] - 2, -150)
            elif key == ord('m'):
                ch_offsets[2] = min(ch_offsets[2] + 2, 150)
            elif key == ord('b'):
                network_input = not network_input
                cv2.destroyWindow('network_input')
            elif key == ord('v'):
                pad_color = (0, 0, 0) if pad_color != (0, 0, 0) else (255, 255, 255)

            if detection_on:
                r = model.forward(frame, tracker_box=[0, 0, 1, 1])
                _, frame, _ = model.detect_and_draw(r, .4, frame, 0)

            frame_show = sutils.add_text_commands(frame, [
                "WASD: move (%d,%d)" % (pi_x, pi_y), "XYZT: scale (%d,%d)" % (pi_w, pi_h),
                "QE: rotate (%d)" % rot_angle,
                "F: perspective", "78: exposure (%d)" % exposure_v, "12: brightness offset (%d)" % brightness_offset,
                "3456P: run/stop detection", "9: interp (%s)" % interpolations[interpolation],
                "0: reload", "UI,JK,NM: BGR changes %s" % ch_offsets,
                "model %s" % type(model).__name__, "detect: %s" % detection_on,
                "H: hide (%s)" % hide,
                "V: pad (%s,%s,%s)" % pad_color, "C: confirm"])

            cv2.imshow("camera-input", frame_show)
            cv2.moveWindow("camera-input", 20, 20)

            if hide:
                window_content = np.zeros_like(window_content)

            cv2.imshow("check", cv2.resize(_adj, (512, 512), interpolations[interpolation]))
            if network_input:
                cv2.imshow("network_input", r["input"])

            cv2.imshow("projection", window_content)
            cv2.moveWindow("projection", main_screen[0], 0)

    except KeyboardInterrupt as e:
        print("Stopping, but saving;)")
        cv2.destroyAllWindows()

    to_save = dict()
    to_save["projection_window_atk"] = dict()
    to_save["projection_window_atk"]["pi_w"] = pi_w
    to_save["projection_window_atk"]["pi_h"] = pi_h
    to_save["projection_window_atk"]["pi_x"] = pi_x
    to_save["projection_window_atk"]["pi_y"] = pi_y
    to_save["projection_window_atk"]["rot_angle"] = rot_angle
    to_save["projection_window_atk"]["pts"] = pts.flatten().tolist()
    to_save["projection_window_atk"]["brightness_offset"] = int(brightness_offset)
    to_save["projection_window_atk"]["interpolation"] = interpolation

    params.update(to_save)

    json.dump(params, open(os.path.join(alignment_folder, "params.json"), "w"), indent=2)







