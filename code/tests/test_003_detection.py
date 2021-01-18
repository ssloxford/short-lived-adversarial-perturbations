import unittest
import os
import subprocess


class TestDetection(unittest.TestCase):
    """
    Tests the execution of the detect.py script for all models
    """

    def _all_files_exists(self, filepaths):
        for fpath in filepaths:
            self.assertEqual(os.path.exists(fpath), True, "Not found %s" % fpath)

    def _check_model_exists(self, models):
        models_folder = os.path.join("/", "home", "data", "models")
        for model in models:

            if model == "yolov3":
                files = [
                    os.path.join(models_folder, "yolov3", "coco.names"),
                    os.path.join(models_folder, "yolov3", "yolo_anchors.txt"),
                    os.path.join(models_folder, "yolov3", "model_v_batch.ckpt.data-00000-of-00001"),
                    os.path.join(models_folder, "yolov3", "model_v_batch.ckpt.index"),
                    os.path.join(models_folder, "yolov3", "model_v_batch.ckpt.meta"),
                ]
                self._all_files_exists(files)

            if model == "maskrcnn":
                files = [
                    os.path.join(models_folder, "maskrcnn", "maskrcnn.data-00000-of-00001"),
                    os.path.join(models_folder, "maskrcnn", "maskrcnn.index"),
                    os.path.join(models_folder, "maskrcnn", "maskrcnn.meta")
                ]
                self._all_files_exists(files)

            if model == "gtsrbcnn_cvpr18":
                files = [
                    os.path.join(models_folder, "gtsrbcnn", "cvpr18", "classes_to_sign_descr.csv"),
                    os.path.join(models_folder, "gtsrbcnn", "cvpr18", "model_best_test.data-00000-of-00001"),
                    os.path.join(models_folder, "gtsrbcnn", "cvpr18", "model_best_test.index"),
                    os.path.join(models_folder, "gtsrbcnn", "cvpr18", "model_best_test.meta"),
                ]
                self._all_files_exists(files)
            if model == "gtsrbcnn_cvpr18iyswim":
                files = [
                    os.path.join(models_folder, "gtsrbcnn", "cvpr18iyswim", "classes_to_sign_descr.csv"),
                    os.path.join(models_folder, "gtsrbcnn", "cvpr18iyswim", "model.data-00000-of-00001"),
                    os.path.join(models_folder, "gtsrbcnn", "cvpr18iyswim", "model.index"),
                    os.path.join(models_folder, "gtsrbcnn", "cvpr18iyswim", "model.meta"),
                ]
                self._all_files_exists(files)
            if model == "gtsrbcnn_usenix21":
                files = [
                    os.path.join(models_folder, "gtsrbcnn", "usenix21", "classes_to_sign_descr.csv"),
                    os.path.join(models_folder, "gtsrbcnn", "usenix21", "gtsrbcnn_scratch.data-00000-of-00001"),
                    os.path.join(models_folder, "gtsrbcnn", "usenix21", "gtsrbcnn_scratch.index"),
                    os.path.join(models_folder, "gtsrbcnn", "usenix21", "gtsrbcnn_scratch.meta"),
                ]
                self._all_files_exists(files)
            if model == "gtsrbcnn_usenix21adv":
                files = [
                    os.path.join(models_folder, "gtsrbcnn", "usenix21adv", "classes_to_sign_descr.csv"),
                    os.path.join(models_folder, "gtsrbcnn", "usenix21adv", "gtsrbcnn_adv.data-00000-of-00001"),
                    os.path.join(models_folder, "gtsrbcnn", "usenix21adv", "gtsrbcnn_adv.index"),
                    os.path.join(models_folder, "gtsrbcnn", "usenix21adv", "gtsrbcnn_adv.meta"),
                ]
                self._all_files_exists(files)

            if model == "lisacnn_cvpr18":
                files = [
                    os.path.join(models_folder, "lisacnn", "cvpr18", "classes_to_sign_descr.csv"),
                    os.path.join(models_folder, "lisacnn", "cvpr18", "all_r_ivan.ckpt.data-00000-of-00001"),
                    os.path.join(models_folder, "lisacnn", "cvpr18", "all_r_ivan.ckpt.index"),
                    os.path.join(models_folder, "lisacnn", "cvpr18", "all_r_ivan.ckpt.meta"),
                ]
                self._all_files_exists(files)
            if model == "lisacnn_cvpr18iyswim":
                files = [
                    os.path.join(models_folder, "lisacnn", "cvpr18iyswim", "classes_to_sign_descr.csv"),
                    os.path.join(models_folder, "lisacnn", "cvpr18iyswim", "model.data-00000-of-00001"),
                    os.path.join(models_folder, "lisacnn", "cvpr18iyswim", "model.index"),
                    os.path.join(models_folder, "lisacnn", "cvpr18iyswim", "model.meta"),
                ]
                self._all_files_exists(files)
            if model == "lisacnn_usenix21":
                files = [
                    os.path.join(models_folder, "lisacnn", "usenix21", "classes_to_sign_descr_slap.csv"),
                    os.path.join(models_folder, "lisacnn", "usenix21", "lisacnn_scratch.data-00000-of-00001"),
                    os.path.join(models_folder, "lisacnn", "usenix21", "lisacnn_scratch.index"),
                    os.path.join(models_folder, "lisacnn", "usenix21", "lisacnn_scratch.meta"),
                ]
                self._all_files_exists(files)
            if model == "lisacnn_usenix21adv":
                files = [
                    os.path.join(models_folder, "lisacnn", "usenix21adv", "classes_to_sign_descr_slap.csv"),
                    os.path.join(models_folder, "lisacnn", "usenix21adv", "lisacnn_adv.data-00000-of-00001"),
                    os.path.join(models_folder, "lisacnn", "usenix21adv", "lisacnn_adv.index"),
                    os.path.join(models_folder, "lisacnn", "usenix21adv", "lisacnn_adv.meta"),
                ]
                self._all_files_exists(files)

    def test_main(self):
        self._check_model_exists([
            "yolov3", "maskrcnn", "gtsrbcnn_cvpr18", "gtsrbcnn_cvpr18iyswim", "gtsrbcnn_usenix21",
            "gtsrbcnn_usenix21adv", "lisacnn_cvpr18", "lisacnn_cvpr18iyswim", "lisacnn_usenix21",
            "lisacnn_usenix21adv"])
        p = [
            {"-n0": "maskrcnn"},
            {"-n0": "gtsrbcnn", "-n1": "cvpr18"},
            {"-n0": "gtsrbcnn", "-n1": "cvpr18iyswim"},
            {"-n0": "gtsrbcnn", "-n1": "usenix21"},
            {"-n0": "gtsrbcnn", "-n1": "usenix21adv"},
            {"-n0": "lisacnn", "-n1": "cvpr18"},
            {"-n0": "lisacnn", "-n1": "cvpr18iyswim"},
            {"-n0": "lisacnn", "-n1": "usenix21"},
            {"-n0": "lisacnn", "-n1": "usenix21adv"},
            {"-n0": "yolov3"},
        ]
        input_filepath = "/home/data/test_run/yolov3_ae.jpeg"
        output_filepath = "/home/code/classifiers/tmp/yolov3_ae.jpeg"
        os.remove(output_filepath) if os.path.exists(output_filepath) else None
        output_files = [output_filepath]

        for config in p:
            additional_args = [[k, v] for k, v in config.items()]
            additional_args = [item for sublist in additional_args for item in sublist]
            cmd = [
                "python",
                "/home/code/classifiers/detect.py",
                "--roi", "100", "200", "300", "400",
                "-f", input_filepath
            ] + additional_args

            print(" ".join(cmd))
            subprocess.run(cmd)
            fps = [(of % config) for of in output_files]
            self._all_files_exists(fps)


