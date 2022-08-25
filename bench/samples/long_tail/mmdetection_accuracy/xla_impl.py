# Copyright(c) OpenMMLab. All Rights Reserved.
# Copied from
import tensorflow as tf
import numpy as np
from bench.core.executer import Executer


def accuracy(pred, target, topk=1, thresh=None):
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    if pred.shape[0] == 0:
        accu = [tf.zeros(tf.shape(pred)) for i in range(len(topk))]
        return accu[0] if return_single else accu
    pred_value, pred_label = tf.math.top_k(pred, maxk) 
    pred_label = tf.transpose(pred_label)  # transpose to shape (maxk, N)
    correct = tf.equal(tf.cast(tf.reshape(target, (1, -1)), tf.int32), pred_label)
    if thresh is not None:
        # Only prediction values larger than thresh are counted as correct
        correct = tf.bitwise.bitwise_and(correct, tf.transpose(tf.math.greater(pred_value, thresh)))
    res = []
    for k in topk:
        correct_k = tf.reduce_sum(tf.cast(tf.reshape(correct[:k], [-1]), tf.float32), axis=0, keepdims=True)
        res.append(tf.multiply(correct_k, 100.0 / pred.shape[0]))
    return res[0] if return_single else res


def args_adaptor(np_args):
    output = tf.convert_to_tensor(np_args[0], tf.float32)
    target = tf.convert_to_tensor(np_args[1], tf.float32)

    return [output, target, 1]


def executer_creator():
    return Executer(accuracy, args_adaptor)
