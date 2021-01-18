import unittest
from shutil import copyfile, rmtree
import os
import subprocess


class TestProjectionModel(unittest.TestCase):
    """
    Tests the execution of the projection_model.py script
    """

    def _setup(self, exp_id, lux, object_id, files_to_delete):
        for f in files_to_delete:
            if os.path.isfile(f):
                os.remove(f)
        exp_folder = os.path.join("/", "home", "data", exp_id,)
        profiling_folder = os.path.join(exp_folder, "profile", str(lux), object_id)
        os.makedirs(profiling_folder, exist_ok=True)
        triples_file = os.path.join(profiling_folder, "all_triples.csv")
        self.assertEqual(os.path.isfile(triples_file), True)

    def test_main(self):
        exp_id = "test_run"
        lux = 1234
        object_ids = ["stop_sign", "give_way", "bottle"]
        exp_folder = os.path.join("/", "home", "data", exp_id,)

        for object_id in object_ids:

            output_pm_fpath = os.path.join(exp_folder, "projection_model", str(lux), object_id, "projection_model.json")
            output_params_fpath = os.path.join(exp_folder, "projection_model", str(lux), object_id, "params.json")

            self._setup(exp_id=exp_id, lux=lux, object_id=object_id, files_to_delete=[output_params_fpath, output_pm_fpath])
            cmd = [
                "python",
                "/home/code/projector/projection_model.py",
                "-id", exp_id,
                "-pm", str(lux),
                "-ob", object_id,
                "--n_epochs", "51",  # only run few
                "--test_network"
            ]
            print(" ".join(cmd))
            subprocess.run(cmd)

            self.assertEqual(os.path.isfile(output_pm_fpath), True, "File not found %s" % output_pm_fpath)
            self.assertEqual(os.path.isfile(output_params_fpath), True, "File not found %s" % output_pm_fpath)


