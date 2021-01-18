import tensorflow as tf
import unittest


class TestGPUAvailable(unittest.TestCase):
    """
    Checks whether a GPU is available
    """

    def test_main(self):
        self.assertTrue(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))



