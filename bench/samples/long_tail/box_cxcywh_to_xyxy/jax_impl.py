# Copyright (c) OpenComputeLab. All Rights Reserved.
import jax.numpy as jnp
from jax import device_put
from bench.core.executer import Executer


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = (x[..., i] for i in range(4))
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return jnp.stack(b, axis=-1)


def args_adaptor(np_args):
    x = device_put(np_args[0])
    return [x]


def executer_creator():
    return Executer(box_cxcywh_to_xyxy, args_adaptor)
