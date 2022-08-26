# Copyright (c) OpenComputeLab. All Rights Reserved.

import tensorflow as tf
import numpy as np
from bench.core.executer import Executer


def label_aware_attention(keys, query):
    """label_aware_attention
    """
    weight = tf.reduce_sum(keys * query, axis=-1)
    weight = tf.pow(weight, 2)  # [x,k_max,1]
    weight = tf.math.softmax(weight, 0)
    output = tf.reduce_sum(keys * weight, axis=1)
    return output, weight


def args_adaptor(np_args):
    keys = tf.convert_to_tensor(np_args[0])
    query = tf.convert_to_tensor(np_args[1])
    return [keys, query]


def executer_creator():
    return Executer(label_aware_attention, args_adaptor)
