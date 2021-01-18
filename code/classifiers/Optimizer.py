import subprocess
import argparse
import yaml


class Optimizer(object):

    def __init__(self, *args, **kwargs):
        super(Optimizer, self).__init__()

    def mandatory_feeds(self, **kwargs):
        return {}

    def adjust_input_for_plot(self, input):
        return input