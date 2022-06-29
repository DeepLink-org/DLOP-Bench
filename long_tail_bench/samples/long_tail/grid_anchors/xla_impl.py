import tensorflow as tf
import numpy as np
from long_tail_bench.core.executer import Executer


def meshgrid(x, y, row_major=True):
    # xx = x.repeat(len(y))
    xx = tf.repeat(x, len(y), 0)
    # yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
    yy = tf.reshape(y, [-1, 1])
    yy = tf.repeat(yy, len(x), 1)
    yy = tf.reshape(yy, [-1])
    if row_major:
        return xx, yy
    else:
        return yy, xx


def grid_anchors(base_anchors, featmap_size, stride):
    feat_h, feat_w = featmap_size
    # shift_x = tf.arange(0, feat_w) * stride
    # shift_y = tf.arange(0, feat_h) * stride
    shift_x = tf.range(0, feat_w, 1) * stride
    shift_y = tf.range(0, feat_h, 1) * stride
    shift_xx, shift_yy = meshgrid(shift_x, shift_y)
    shifts = tf.stack([shift_xx, shift_yy, shift_xx, shift_yy], axis=-1)
    # shifts = shifts.type_as(base_anchors)
    shifts = tf.cast(shifts, base_anchors.dtype)
    all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
    # all_anchors = all_anchors.reshape(-1, 4)
    all_anchors = tf.reshape(all_anchors, [-1, 4])
    return all_anchors


def args_adaptor(np_args):
    base_anchors = tf.convert_to_tensor(np_args[0], tf.float32)

    return [base_anchors, np_args[1], np_args[2]]


def executer_creator():
    return Executer(grid_anchors, args_adaptor)
