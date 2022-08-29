# Copyright (c) OpenComputeLab. All Rights Reserved.
import tensorflow as tf
import numpy as np
from bench.core.executer import Executer


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = tf.unstack(x, axis=-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return tf.stack(b, axis=-1)


def args_adaptor(np_args):
    x = tf.convert_to_tensor(np_args[0], tf.float32)
    return [x]


def executer_creator():
    return Executer(box_cxcywh_to_xyxy, args_adaptor)
