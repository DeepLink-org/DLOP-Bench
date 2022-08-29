# Copyright (c) OpenComputeLab. All Rights Reserved.
import tensorflow as tf
import numpy as np
from bench.core.executer import Executer


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def args_adaptor(np_args):
    boxes = tf.convert_to_tensor(np_args[0], tf.float32)
    return [boxes]


def executer_creator():
    return Executer(box_area, args_adaptor)
