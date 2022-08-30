# Copyright (c) OpenComputeLab. All Rights Reserved.
import tensorflow as tf
import numpy as np
from bench.core.executer import Executer


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

def quality_focal_loss(pred, target, beta=2.0):
    assert len(target) == 2, """target for QFL must be a tuple of two elements,
        including category label and quality label, respectively"""
    # label denotes the category id, score denotes the quality score
    label, score = target

    # negatives are supervised by 0 quality score
    pred_sigmoid = tf.math.sigmoid(pred)
    scale_factor = pred_sigmoid
    zerolabel = tf.zeros(tf.shape(pred))
    loss = tf.reshape(cross_entropy(pred, zerolabel), (-1, 1)) * tf.math.pow(scale_factor, beta)
    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    bg_class_ind = pred.shape[1]
    pos = tf.math.logical_and(tf.math.greater_equal(label, 0), tf.math.less(label, bg_class_ind))
    pos = tf.squeeze(tf.experimental.numpy.nonzero(pos), axis=1)
    pos_label = tf.cast(label[pos], dtype=tf.int64)
    # positives are supervised by bbox quality (IoU) score
    scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
    loss[pos, pos_label] = cross_entropy(pred[pos, pos_label], score[pos]) * tf.math.pow(tf.math.abs(scale_factor), beta)
    loss = tf.experimental.numpy.sum(loss, axis=1, keepdims=None)
    return loss


def args_adaptor(np_args):
    pred = tf.convert_to_tensor(np_args[0])
    target_0 = tf.convert_to_tensor(np_args[1])
    target_1 = tf.convert_to_tensor(np_args[2])
    # pred.requires_grad = True

    return [pred, (target_0, target_1)]


def executer_creator():
    return Executer(quality_focal_loss, args_adaptor)
