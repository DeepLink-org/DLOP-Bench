# Copyright (c) OpenComputeLab. All Rights Reserved.
import tensorflow as tf
import numpy as np
from bench.core.executer import Executer


def legacy_bbox2delta(proposals,
                      gt,
                      means=(0., 0., 0., 0.),
                      stds=(1., 1., 1., 1.)):
    """Compute deltas of proposals w.r.t. gt in the MMDet V1.x manner.

    We usually compute the deltas of x, y, w, h of proposals w.r.t ground
    truth bboxes to get regression target.
    This is the inverse function of `delta2bbox()`

    Args:
        proposals (Tensor): Boxes to be transformed, shape (N, ..., 4)
        gt (Tensor): Gt bboxes to be used as base, shape (N, ..., 4)
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates

    Returns:
        Tensor: deltas with shape (N, 4), where columns represent dx, dy,
            dw, dh.
    """

    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0] + 1.0
    ph = proposals[..., 3] - proposals[..., 1] + 1.0

    gx = (gt[..., 0] + gt[..., 2]) * 0.5
    gy = (gt[..., 1] + gt[..., 3]) * 0.5
    gw = gt[..., 2] - gt[..., 0] + 1.0
    gh = gt[..., 3] - gt[..., 1] + 1.0

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = tf.math.log(gw / pw)
    dh = tf.math.log(gh / ph)
    deltas = tf.stack([dx, dy, dw, dh], axis=-1)

    means = np.array(means)
    means = tf.expand_dims(means, 0)
    stds = np.array(stds)
    stds = tf.expand_dims(stds, 0)
    means = tf.cast(means, dtype = tf.float32)
    stds = tf.cast(stds, dtype = tf.float32)
    deltas = deltas - means
    deltas = deltas / stds
    return deltas

@tf.function(experimental_compile=True)
def fast_bbox2delta(proposals,
                      gt,
                      means=(0., 0., 0., 0.),
                      stds=(1., 1., 1., 1.)):
    """Compute deltas of proposals w.r.t. gt in the MMDet V1.x manner.

    We usually compute the deltas of x, y, w, h of proposals w.r.t ground
    truth bboxes to get regression target.
    This is the inverse function of `delta2bbox()`

    Args:
        proposals (Tensor): Boxes to be transformed, shape (N, ..., 4)
        gt (Tensor): Gt bboxes to be used as base, shape (N, ..., 4)
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates

    Returns:
        Tensor: deltas with shape (N, 4), where columns represent dx, dy,
            dw, dh.
    """

    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0] + 1.0
    ph = proposals[..., 3] - proposals[..., 1] + 1.0

    gx = (gt[..., 0] + gt[..., 2]) * 0.5
    gy = (gt[..., 1] + gt[..., 3]) * 0.5
    gw = gt[..., 2] - gt[..., 0] + 1.0
    gh = gt[..., 3] - gt[..., 1] + 1.0

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = tf.math.log(gw / pw)
    dh = tf.math.log(gh / ph)
    deltas = tf.stack([dx, dy, dw, dh], axis=-1)

    means = np.array(means)
    means = tf.expand_dims(means, 0)
    stds = np.array(stds)
    stds = tf.expand_dims(stds, 0)
    means = tf.cast(means, dtype = tf.float32)
    stds = tf.cast(stds, dtype = tf.float32)
    deltas = deltas - means
    deltas = deltas / stds
    return deltas

def args_adaptor(np_args):
    proposals = tf.convert_to_tensor(np_args[0], tf.float32)
    gt = tf.convert_to_tensor(np_args[1], tf.float32)

    return [proposals, gt, np_args[2], np_args[3]]


def executer_creator():
    return Executer(legacy_bbox2delta, args_adaptor)
