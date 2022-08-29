# Copyright (c) OpenComputeLab. All Rights Reserved.
import tensorflow as tf
import numpy as np
from bench.core.executer import Executer


def yolo_encode(bboxes, gt_bboxes, stride, eps):
    """Get box regression transformation deltas that can be used to
    transform the ``bboxes`` into the ``gt_bboxes``.

    Args:
        bboxes (torch.Tensor): Source boxes, e.g., anchors.
        gt_bboxes (torch.Tensor): Target of the transformation, e.g.,
            ground-truth boxes.
        stride (torch.Tensor | int): Stride of bboxes.

    Returns:
        torch.Tensor: Box transformation deltas
    """

    x_center_gt = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) * 0.5
    y_center_gt = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) * 0.5
    w_gt = gt_bboxes[..., 2] - gt_bboxes[..., 0]
    h_gt = gt_bboxes[..., 3] - gt_bboxes[..., 1]
    x_center = (bboxes[..., 0] + bboxes[..., 2]) * 0.5
    y_center = (bboxes[..., 1] + bboxes[..., 3]) * 0.5
    w = bboxes[..., 2] - bboxes[..., 0]
    h = bboxes[..., 3] - bboxes[..., 1]
    w_target = tf.math.log(w_gt / w)
    w_target = tf.clip_by_value(w_target, eps, w_target)
    h_target = tf.math.log(h_gt / h)
    h_target = tf.clip_by_value(h_target, eps, h_target)
    x_center_target = ((x_center_gt - x_center) / stride + 0.5)
    x_center_target = tf.clip_by_value(x_center_target, eps, 1 - eps)
    y_center_target = ((y_center_gt - y_center) / stride + 0.5)
    y_center_target = tf.clip_by_value(y_center_target, eps, 1 - eps)
    encoded_bboxes = tf.stack(
        [x_center_target, y_center_target, w_target, h_target], axis=-1)
    return encoded_bboxes


def args_adaptor(np_args):
    bboxes = tf.convert_to_tensor(np_args[0], tf.float32)
    gt_bboxes = tf.convert_to_tensor(np_args[1], tf.float32)
    stride = tf.convert_to_tensor(np_args[2], tf.float32)
    eps = 1e-6

    return [bboxes, gt_bboxes, stride, eps]


def executer_creator():
    return Executer(yolo_encode, args_adaptor)
