# Copyright (c) OpenComputeLab. All Rights Reserved.

import jax.numpy as jnp
from jax import device_put
from bench.core.executer import Executer

def meshgrid(x, y, row_major=True):
    xx = x.repeat(len(y))
    yy = y.reshape(-1, 1).tile(len(x)).reshape(-1) #np.tile <- torch.repeat
    if row_major:
        return xx, yy
    else:
        return yy, xx

def grid_anchors(base_anchors, featmap_size, stride):

    feat_h, feat_w = featmap_size
    shift_x = jnp.arange(0, feat_w) * stride
    shift_y = jnp.arange(0, feat_h) * stride
    shift_xx, shift_yy = meshgrid(shift_x, shift_y)
    shifts = jnp.stack([shift_xx, shift_yy, shift_xx, shift_yy], axis=-1)
    # shifts = shifts.type_as(base_anchors) #jax cannot handle type_as

    all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
    all_anchors = all_anchors.reshape(-1, 4) #np.reshape <- torch.view
    return all_anchors

def args_adaptor(np_args):
    base_anchors = device_put(np_args[0])

    return [base_anchors, np_args[1], np_args[2]]


def executer_creator():
    return Executer(grid_anchors, args_adaptor)
