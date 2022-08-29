# Copyright (c) OpenComputeLab. All Rights Reserved.
import tensorflow as tf
from bench.core.executer import Executer


tf.compat.v1.enable_eager_execution()
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))


def accuracy(output, target, topk=(1,), raw=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    
    maxk = max(topk)
    _, pred = tf.math.top_k(output, maxk, True) 
    pred = tf.cast(tf.transpose(pred), tf.int64)   
    correct = tf.equal(tf.reshape(target, (1, -1)), pred)

    res = []
    for k in topk:
        correct_k = tf.reduce_sum(tf.cast(tf.reshape(correct[:k], [-1]), tf.float32), axis=0, keepdims=True)
        if raw:
            res.append(correct_k)
        else:
            res.append(tf.multiply(correct_k, 100.0 / target.shape[0]))
    return res


def args_adaptor(np_args):
    output = tf.cast(tf.convert_to_tensor(np_args[0]), tf.float32)
    target = tf.convert_to_tensor(np_args[1])
    return [output, target, (1, 5)]


def executer_creator():
    return Executer(accuracy, args_adaptor)
