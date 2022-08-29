# Copyright (c) OpenComputeLab. All Rights Reserved.

import tensorflow as tf
import numpy as np
from bench.core.executer import Executer


def xyxy2xywh(boxes, stacked=False):
    """(x1, y1, x2, y2) -> (x, y, w, h)"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    if stacked:
        return tf.stack([cx, cy, w, h], dim=1)
    else:
        return cx, cy, w, h


def bbox2offset(boxes, gt_boxes, weights=(1.0, 1.0, 1.0, 1.0)):
    """
    Inverse transform that computes target bounding-box regression deltas
    given proposal boxes and ground-truth boxes. The weights argument should be
    a 4-tuple of multiplicative weights that are applied to the regression
    target.

    In older versions of this code (and in py-faster-rcnn), the weights were
    set such that the regression deltas would have unit standard deviation on
    the training dataset. Presently, rather than computing these statistics
    exactly, we use a fixed set of weights (10., 10., 5., 5.) by default.
    These are approximately the weights one would get from COCO using the
    previous unit stdev heuristic.
    """
    assert boxes.shape[0] == gt_boxes.shape[0]
    ex_ctr_x, ex_ctr_y, ex_widths, ex_heights = xyxy2xywh(boxes)
    gt_ctr_x, gt_ctr_y, gt_widths, gt_heights = xyxy2xywh(gt_boxes)

    wx, wy, ww, wh = weights
    offset_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    offset_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    offset_dw = ww * tf.math.log(gt_widths / ex_widths)
    offset_dh = wh * tf.math.log(gt_heights / ex_heights)
    offset = tf.stack((offset_dx, offset_dy, offset_dw, offset_dh), axis=1)
    return offset


def args_adaptor(np_args):
    boxes = tf.convert_to_tensor(np_args[0], tf.float32)
    gt = tf.convert_to_tensor(np_args[1], tf.float32)
    weights = [1.0, 1.0, 1.0, 1.0]
    return [boxes, gt, weights]


def executer_creator():
    return Executer(bbox2offset, args_adaptor)
