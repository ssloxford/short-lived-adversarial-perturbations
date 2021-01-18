import unittest
from shutil import copyfile, rmtree
import os
import subprocess


class TestDefences(unittest.TestCase):
    """
    Tests the execution of defences/adv_train.py and defences/sentinet.py
    """

    def test_adversarial_train(self):

        p = [
            ["-m", "lisacnn"],
            ["-m", "lisacnn", "-a"],
            ["-m", "gtsrbcnn"],
            ["-m", "gtsrbcnn", "-a"],
        ]

        for config in p:

            model_name = config[1] + ("" if "-a" not in config else "_adv") + ".h5"
            output_h5_file = os.path.join("/", "home", "code", "defences", "tmp", model_name)

            cmd = [
                "python",
                "/home/code/defences/adv_train.py",
            ] + config + ["--training_epochs", "1"]

            print(" ".join(cmd))
            subprocess.run(cmd)

            self.assertEqual(os.path.isfile(output_h5_file), True, "File not found %s" % output_h5_file)

    def test_sentinet(self):

        p = [
            ["-m", "gtsrbcnn_cvpr18"],
            ["-m", "lisacnn_cvpr18"]
        ]

        for config in p:
            output_adv_file = os.path.join("/", "home", "data", "test_run", "sentinet", "adversarial_results.csv")
            output_benign_file = os.path.join("/", "home", "data", "test_run", "sentinet", "benign_results.csv")
            os.remove(output_adv_file) if os.path.exists(output_adv_file) else None
            os.remove(output_benign_file) if os.path.exists(output_benign_file) else None

            cmd = [
                "python",
                "/home/code/defences/sentinet.py",
                "-t", "/home/data/test_run/sentinet/reference/",
                "-b", "/home/data/test_run/sentinet/benign/",
                "-a", "/home/data/test_run/sentinet/adversarial/",
                "-o", "/home/data/test_run/sentinet/"
            ] + config

            print(" ".join(cmd))

            subprocess.run(cmd)

            self.assertEqual(os.path.isfile(output_adv_file), True, "File not found %s" % output_adv_file)
            self.assertEqual(os.path.isfile(output_benign_file), True, "File not found %s" % output_benign_file)


