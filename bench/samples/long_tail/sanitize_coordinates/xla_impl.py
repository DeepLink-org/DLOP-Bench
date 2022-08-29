# Copyright (c) OpenComputeLab. All Rights Reserved.
import tensorflow as tf
import numpy as np
from bench.core.executer import Executer


def sanitize_coordinates(_x1, _x2, img_size, padding=0, cast=True):
    """
    Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0,
    and x2 <= image_size.
    Also converts from relative to absolute coordinates
    and casts the results to long tensors.

    If cast is false, the result won't be cast to longs.
    Warning: this does things in-place behind the scenes so copy if necessary.
    """
    _x1 *= img_size
    _x2 *= img_size
    if cast:
        _x1 = tf.cast(_x1, tf.int64)
        _x2 = tf.cast(_x2, tf.int64)
    x1 = tf.minimum(_x1, _x2)
    x2 = tf.maximum(_x1, _x2)
    x1 = tf.clip_by_value(x1 - padding, clip_value_min=0, clip_value_max=tf.float32.max)
    x2 = tf.clip_by_value(x2 + padding, clip_value_min=tf.float32.min, clip_value_max=img_size)
    return x1, x2


def args_adaptor(np_args):
    x1 = tf.convert_to_tensor(np_args[0])
    x2 = tf.convert_to_tensor(np_args[1])
    return [x1, x2, 256, 0, True]


def executer_creator():
    return Executer(sanitize_coordinates, args_adaptor)
