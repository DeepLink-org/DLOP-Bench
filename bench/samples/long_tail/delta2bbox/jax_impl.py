# Copyright (c) OpenComputeLab. All Rights Reserved.
import jax.numpy as jnp
from jax import jit, device_put
from bench.core.executer import Executer


def delta2bbox(rois,
               deltas,
               means,
               stds,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
    # if isinstance(deltas, jnp.cuda.HalfTensor):
    #     deltas = deltas.float()
    means = jnp.array(means)
    means = jnp.expand_dims(means, axis=0)
    means = jnp.tile(means, (deltas.shape[0] ,1))
    stds = jnp.array(stds)
    stds = jnp.expand_dims(stds, axis=0)
    stds = jnp.tile(stds, (deltas.shape[0] ,1))
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::4]
    dy = denorm_deltas[:, 1::4]
    dw = denorm_deltas[:, 2::4]
    dh = denorm_deltas[:, 3::4]
    max_ratio = jnp.abs(jnp.log(wh_ratio_clip))
    dw = jnp.clip(dw, -max_ratio, max_ratio)
    dh = jnp.clip(dh, -max_ratio, max_ratio)
    px = ((rois[:, 0] + rois[:, 2]) * 0.5)
    px = jnp.expand_dims(px, axis=-1)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5)
    py = jnp.expand_dims(py, axis=-1)
    pw = (rois[:, 2] - rois[:, 0] + 1.0)
    pw = jnp.expand_dims(pw, axis=-1)
    ph = (rois[:, 3] - rois[:, 1] + 1.0)
    ph = jnp.expand_dims(ph, axis=-1)
    gw = pw * jnp.exp(dw)
    gh = ph * jnp.exp(dh)
    gx = px + pw * dx
    gy = py + ph * dy
    x1 = gx - gw * 0.5 + 0.5
    y1 = gy - gh * 0.5 + 0.5
    x2 = gx + gw * 0.5 - 0.5
    y2 = gy + gh * 0.5 - 0.5
    if max_shape is not None:
        x1 = jnp.clip(x1, 0, max_shape[1] - 1)
        y1 = jnp.clip(y1, 0, max_shape[0] - 1)
        x2 = jnp.clip(x2, 0, max_shape[1] - 1)
        y2 = jnp.clip(y2, 0, max_shape[0] - 1)
    bboxes = jnp.stack([x1, y1, x2, y2], axis=-1)
    bboxes = jnp.reshape(bboxes, deltas.shape)
    return bboxes


def args_adaptor(np_args):
    rois = device_put(np_args[0])
    deltas = device_put(np_args[1])

    means = device_put(np_args[2])
    stds = device_put(np_args[3])
    max_shape = device_put(np_args[4])

    return [rois, deltas, means, stds, max_shape]


def executer_creator():
    return Executer(delta2bbox, args_adaptor)
