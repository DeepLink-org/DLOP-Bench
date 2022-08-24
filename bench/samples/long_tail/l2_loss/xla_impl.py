import numpy as np
import tensorflow as tf
from bench.core.executer import Executer


def l2_loss(input, target):
    loss = tf.reduce_mean((input - target) * (input - target))
    return loss

def args_generator(np_args):
    output = tf.convert_to_tensor(np_args[0], tf.float32)
    target = tf.convert_to_tensor(np_args[1], tf.float32)
    return [output, target]


def executer_creator():
    return Executer(l2_loss, args_generator)
