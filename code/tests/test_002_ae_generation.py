import unittest
from shutil import copyfile, rmtree
import os
import subprocess


class TestAEGeneration(unittest.TestCase):
    """
    Tests the execution of the projection_model.py script
    """

    def _all_files_exists(self, filepaths):
        for fpath in filepaths:
            self.assertEqual(os.path.exists(fpath), True, "Not found %s" % fpath)

    def _check_object_exists(self, exp_id, objects):
        for object in objects:
            obj_png = os.path.join("/", "home", "data", exp_id, "objects", object+".png")
            self.assertEqual(os.path.isfile(obj_png), True, "File found %s" % obj_png)

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
            if model == "gtsrbcnn_cvpr18":
                files = [
                    os.path.join(models_folder, "gtsrbcnn", "cvpr18", "classes_to_sign_descr.csv"),
                    os.path.join(models_folder, "gtsrbcnn", "cvpr18", "model_best_test.data-00000-of-00001"),
                    os.path.join(models_folder, "gtsrbcnn", "cvpr18", "model_best_test.index"),
                    os.path.join(models_folder, "gtsrbcnn", "cvpr18", "model_best_test.meta"),
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

            if model == "maskrcnn":
                files = [
                    os.path.join(models_folder, "maskrcnn", "maskrcnn.data-00000-of-00001"),
                    os.path.join(models_folder, "maskrcnn", "maskrcnn.index"),
                    os.path.join(models_folder, "maskrcnn", "maskrcnn.meta")
                ]
                self._all_files_exists(files)

    def test_main(self):
        exp_id = "test_run"
        lux = str(1234)

        self._check_object_exists(exp_id, ["stop_sign", "bottle", "give_way"])
        self._check_model_exists(["yolov3", "maskrcnn", "gtsrbcnn_cvpr18", "lisacnn_cvpr18"])

        p = [
            {"-ob": "stop_sign", "-n": "gtsrbcnn_cvpr18"},
            {"-ob": "give_way", "-n": "gtsrbcnn_cvpr18"},
            {"-ob": "stop_sign", "-n": "lisacnn_cvpr18"},
            {"-ob": "give_way", "-n": "lisacnn_cvpr18"},
            {"-ob": "stop_sign", "-n": "maskrcnn"},
            {"-ob": "bottle", "-n": "maskrcnn"},
            {"-ob": "stop_sign", "-n": "yolov3"},
            {"-ob": "bottle", "-n": "yolov3"},
        ]

        exp_folder = os.path.join("/", "home", "data", exp_id,)
        output_folder = os.path.join(exp_folder, "optimize", str(lux))
        rmtree(output_folder, ignore_errors=True)

        output_files = [
            os.path.join(output_folder, "%(-ob)s", "%(-n)s", "params.json"),
            os.path.join(output_folder, "%(-ob)s", "%(-n)s", "_best_projection.png"),
            os.path.join(output_folder, "%(-ob)s", "%(-n)s", "projections.npy"),
            os.path.join(output_folder, "%(-ob)s", "%(-n)s", "inputs"),
            os.path.join(output_folder, "%(-ob)s", "%(-n)s", "traindata"),
            os.path.join(output_folder, "%(-ob)s", "%(-n)s", "visualize"),
        ]

        for object_id in ["stop_sign", "give_way", "bottle"]:
            # check needed files exist
            prj_model_files = [
                os.path.join(exp_folder, "projection_model", lux, object_id, "params.json"),
                os.path.join(exp_folder, "projection_model", lux, object_id, "projection_model.json")
            ]
            self._all_files_exists(prj_model_files)

        for config in p:
            cmd = [
                "python",
                "/home/code/classifiers/optimize.py",
                "-id", exp_id,
                "-pm", lux,
                "-ob", config["-ob"],
                "-n", config["-n"],
                "--n_epochs", "1"  # only run for few epochs
            ]

            print(" ".join(cmd))
            subprocess.run(cmd)
            fps = [(of % config) for of in output_files]
            self._all_files_exists(fps)


