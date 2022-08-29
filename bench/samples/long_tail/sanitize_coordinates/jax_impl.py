# Copyright (c) OpenComputeLab. All Rights Reserved.
import jax.numpy as jnp
from jax import device_put
from bench.core.executer import Executer


def sanitize_coordinates(_x1, _x2, img_size, padding=0, cast=False):
    """
    Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0,
    and x2 <= image_size.
    Also converts from relative to absolute coordinates
    and casts the results to long tensors.

    If cast is false, the result won't be cast to longs.
    Warning: this does things in-place behind the scenes so copy if necessary.
    """
    _x1 *= img_size
    _x2 *= img_size
    # if cast:
    #     _x1 = _x1.long()
    #     _x2 = _x2.long()
    x1 = jnp.minimum(_x1, _x2)
    x2 = jnp.maximum(_x1, _x2)
    x1 = jnp.clip(x1 - padding, 0)
    x2 = jnp.clip(x2 + padding, 0, img_size)
    return x1, x2


def args_adaptor(np_args):
    x1 = device_put(np_args[0])
    x2 = device_put(np_args[1])
    return [x1, x2, 256, 0, True]


def executer_creator():
    return Executer(sanitize_coordinates, args_adaptor)
