# Copyright (c) OpenComputeLab. All Rights Reserved.
import tensorflow as tf
import numpy as np
from bench.core.executer import Executer


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = tf.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = tf.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = tf.clip_by_value(rb - lt, clip_value_min=0, clip_value_max=tf.reduce_max(rb - lt))
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def args_adaptor(np_args):
    boxes1 = tf.convert_to_tensor(np_args[0], tf.float32)
    boxes2 = tf.convert_to_tensor(np_args[1], tf.float32)
    return [boxes1, boxes2]


def executer_creator():
    return Executer(box_iou, args_adaptor)
