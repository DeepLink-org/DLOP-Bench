# Copyright (c) OpenComputeLab. All Rights Reserved.
import tensorflow as tf
import numpy as np
from bench.core.executer import Executer


class SmoothStep():
    def __init__(self, gamma=1.0):
        super(SmoothStep, self).__init__()
        self._lower_bound = -gamma / 2.0
        self._upper_bound = gamma / 2.0
        self._a3 = -2 / (gamma**3)
        self._a1 = 3 / (2 * gamma)
        self._a0 = 0.5

    def forward(self, inputs):
        return tf.where(
            inputs <= self._lower_bound, tf.zeros_like(inputs),
            tf.where(inputs >= self._upper_bound, tf.ones_like(inputs),
                        self._a3 * (inputs**3) + self._a1 * inputs + self._a0))


def args_adaptor(np_args):
    inputs = tf.convert_to_tensor(np_args[0])
    return [inputs]


def executer_creator():
    coder_instance = SmoothStep()
    return Executer(coder_instance.forward, args_adaptor)
