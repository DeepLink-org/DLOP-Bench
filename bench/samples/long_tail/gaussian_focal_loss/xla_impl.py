# Copyright (c) OpenComputeLab. All Rights Reserved.

import tensorflow as tf
import numpy as np
from bench.core.executer import Executer


def gaussian_focal_loss(pred, gaussian_target, alpha=2.0, gamma=4.0):
    eps = 1e-12
    pos_weights = tf.math.equal(gaussian_target, 1)
    neg_weights = tf.math.pow((1 - gaussian_target), gamma)
    pos_loss = -tf.math.log(pred + eps) * tf.math.pow((1 - pred), alpha) * tf.dtypes.cast(pos_weights, dtype=tf.float32)
    neg_loss = -tf.math.log(1 - pred + eps) * tf.math.pow(pred, alpha) * neg_weights
    return pos_loss + neg_loss


def args_adaptor(np_args):
    pred = tf.convert_to_tensor(np_args[0], tf.float32)
    # pred.requires_grad = True
    target = tf.convert_to_tensor(np_args[1], tf.float32)
    return [pred, target]


def executer_creator():
    return Executer(gaussian_focal_loss, args_adaptor)
