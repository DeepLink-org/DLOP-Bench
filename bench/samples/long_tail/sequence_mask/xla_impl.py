# Copyright (c) OpenComputeLab. All Rights Reserved.
import tensorflow as tf
import numpy as np
from bench.core.executer import Executer


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = tf.size(lengths)
    max_len = lengths[0]
    max_len_tensor = tf.range(0, max_len)
    max_len_tensor = tf.repeat(max_len_tensor, batch_size, 0)
    lengths = tf.repeat(lengths, int(max_len), 0)
    lengths =  tf.expand_dims(lengths, 1)
    return tf.less(max_len_tensor, lengths)


def args_adaptor(np_args):
    input0 = tf.convert_to_tensor(np_args[0])
    return [input0]


def executer_creator():
    return Executer(sequence_mask, args_adaptor)
