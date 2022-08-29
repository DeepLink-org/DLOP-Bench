# Copyright (c) OpenComputeLab. All Rights Reserved.

import tensorflow as tf
import numpy as np
from bench.core.executer import Executer


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = tf.unstack(x, axis=-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return tf.stack(b, axis=-1)


def args_adaptor(np_args):
    x = tf.convert_to_tensor(np_args[0], tf.float32)
    return [x]


def executer_creator():
    return Executer(box_xyxy_to_cxcywh, args_adaptor)
