import tensorflow as tf
import numpy as np
from long_tail_bench.core.executer import Executer


def bbox_target_expand(bbox_targets, bbox_weights, labels, bbox_targets_expand, bbox_weights_expand):
    for i in tf.squeeze(tf.where(labels > 0), axis=-1):
        start, end = labels[i] * 4, (labels[i] + 1) * 4
        # bbox_targets_expand[i, start:end] = bbox_targets[i, :]
        # bbox_weights_expand[i, start:end] = bbox_weights[i, :]
        bbox_targets_expand = tf.Variable(bbox_targets_expand)
        bbox_weights_expand = tf.Variable(bbox_weights_expand)
        bbox_targets_expand[:, start:end].assign(bbox_targets[i, :])
        bbox_weights_expand[:, start:end].assign(bbox_weights[i, :])
        bbox_targets_expand = tf.convert_to_tensor(bbox_targets_expand)
        bbox_weights_expand = tf.convert_to_tensor(bbox_weights_expand)
    return bbox_targets_expand, bbox_weights_expand


def args_adaptor(np_args):
    bbox_targets = tf.convert_to_tensor(np_args[0], tf.float32)
    bbox_weights = tf.convert_to_tensor(np_args[1], tf.float32)
    labels = tf.convert_to_tensor(np_args[2])
    bbox_targets_expand = tf.convert_to_tensor(np_args[3], tf.float32)
    bbox_weights_expand = tf.convert_to_tensor(np_args[4], tf.float32)

    return [
        bbox_targets,
        bbox_weights,
        labels,
        bbox_targets_expand,
        bbox_weights_expand,
    ]


def executer_creator():
    return Executer(bbox_target_expand, args_adaptor)
