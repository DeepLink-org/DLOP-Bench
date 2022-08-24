import tensorflow as tf
import numpy as np
from bench.core.executer import Executer


def squash(Z):
    """squash
    """
    vec_squared_norm = tf.reduce_sum(tf.square(Z), axis=-1)
    scalar_factor = vec_squared_norm / \
        (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + 1e-8)
    vec_squashed = scalar_factor * Z
    return vec_squashed


def args_adaptor(np_args):
    input0 = tf.convert_to_tensor(np_args[0])
    return [input0]


def executer_creator():
    return Executer(squash, args_adaptor)
