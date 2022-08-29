# Copyright (c) OpenComputeLab. All Rights Reserved.
import tensorflow as tf
import numpy as np
from bench.core.executer import Executer


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def distribution_focal_loss(pred, label):
    dis_left = tf.dtypes.cast(label, dtype=tf.int64)
    dis_right = dis_left + 1
    weight_left = tf.dtypes.cast(dis_right, dtype=tf.float32) - label
    weight_right = label - tf.dtypes.cast(dis_left, dtype=tf.float32)
    # TODO(limaolin): need check differences between tf.keras.losses.BinaryCrossentropy
    # and pytorch F.cross_entropy
    cross_entropy = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    loss = cross_entropy(pred, dis_left) * weight_left \
        + cross_entropy(pred, dis_right) * weight_right
    return loss


def args_adaptor(np_args):
    pred = tf.convert_to_tensor(np_args[0], tf.float32)
    # pred.requires_grad = True
    target = tf.convert_to_tensor(np_args[1], tf.float32)
    return [pred, target]


def executer_creator():
    return Executer(distribution_focal_loss, args_adaptor)
