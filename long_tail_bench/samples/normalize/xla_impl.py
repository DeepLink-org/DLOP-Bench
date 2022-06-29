import tensorflow as tf
import numpy as np
from long_tail_bench.core.executer import Executer


def normalize(src, mean, scale):
    # RGB to BGR
    dup = tf.Variable(src)
    dup[:, :, 0].assign(src[:, :, 2])
    dup[:, :, 2].assign(src[:, :, 0])
    dup = tf.convert_to_tensor(dup)
    return (dup - mean) * scale


def args_adaptor(np_args):
    img = tf.convert_to_tensor(np_args[0])
    mean = tf.convert_to_tensor(np_args[1])
    scale = tf.convert_to_tensor(np_args[2])

    return [img, mean, scale]


def executer_creator():
    return Executer(normalize, args_adaptor)
