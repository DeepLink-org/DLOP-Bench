# Copyright (c) OpenComputeLab. All Rights Reserved.
import tensorflow as tf
import numpy as np
from bench.core.executer import Executer


def map_roi_levels(rois, num_levels=4, finest_scale=56):
    """Map rois to corresponding feature levels by scales.

    - scale < finest_scale * 2: level 0
    - finest_scale * 2 <= scale < finest_scale * 4: level 1
    - finest_scale * 4 <= scale < finest_scale * 8: level 2
    - scale >= finest_scale * 8: level 3

    Args:
        rois (Tensor): Input RoIs, shape (k, 5).
        num_levels (int): Total level number.

    Returns:
        Tensor: Level index (0-based) of each RoI, shape (k, )
    """
    scale = tf.sqrt(
        (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
    target_lvls = tf.floor(tf.math.log(scale / finest_scale + 1e-6) / tf.math.log(2.))
    target_lvls = tf.clip_by_value(target_lvls, clip_value_min=0, clip_value_max=num_levels - 1)

    return target_lvls

@tf.function(experimental_compile=True)
def fast_map_roi_levels(rois, num_levels=4, finest_scale=56):
    """Map rois to corresponding feature levels by scales.

    - scale < finest_scale * 2: level 0
    - finest_scale * 2 <= scale < finest_scale * 4: level 1
    - finest_scale * 4 <= scale < finest_scale * 8: level 2
    - scale >= finest_scale * 8: level 3

    Args:
        rois (Tensor): Input RoIs, shape (k, 5).
        num_levels (int): Total level number.

    Returns:
        Tensor: Level index (0-based) of each RoI, shape (k, )
    """
    scale = tf.sqrt(
        (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
    target_lvls = tf.floor(tf.math.log(scale / finest_scale + 1e-6) / tf.math.log(2.))
    target_lvls = tf.clip_by_value(target_lvls, clip_value_min=0, clip_value_max=num_levels - 1)

    return target_lvls


def args_adaptor(np_args):
    rois = tf.convert_to_tensor(np_args[0], tf.float32)
    return [rois, np_args[1], np_args[2]]


def executer_creator():
    return Executer(map_roi_levels, args_adaptor)
