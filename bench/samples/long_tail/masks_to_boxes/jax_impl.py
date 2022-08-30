# Copyright (c) OpenComputeLab. All Rights Reserved.

import jax.numpy as jnp
from jax import device_put
from bench.core.executer import Executer


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.size == 0:
        return jnp.zeros((0, 4))

    h, w = masks.shape[-2:]

    y = jnp.arange(0, h)
    x = jnp.arange(0, w)
    y, x = jnp.meshgrid(y, x)

    x_mask = (masks * jnp.expand_dims(x, 0))
    x_max = jnp.max(jnp.reshape(x_mask, (x_mask.shape[0], -1)), axis=-1)
    x_min = jnp.min(jnp.reshape(jnp.where(masks.astype('bool_'), x_mask, 1e8), (x_mask.shape[0], -1)), axis=-1)

    y_mask = (masks * jnp.expand_dims(y, 0))
    y_max = jnp.max(jnp.reshape(y_mask, (y_mask.shape[0], -1)), axis=-1)
    y_min = jnp.min(jnp.reshape(jnp.where(masks.astype('bool_'), y_mask, 1e8), (y_mask.shape[0], -1)), axis=-1)

    return jnp.stack([x_min, y_min, x_max, y_max], 1)


def args_adaptor(np_args):
    masks = device_put(np_args[0])

    return [masks]


def executer_creator():
    return Executer(masks_to_boxes, args_adaptor)
