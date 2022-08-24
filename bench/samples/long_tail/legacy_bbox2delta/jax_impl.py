from jax import numpy as np
import jax.numpy as jnp
from jax import device_put
from bench.core.executer import Executer


def legacy_bbox2delta(proposals,
                      gt,
                      means=(0., 0., 0., 0.),
                      stds=(1., 1., 1., 1.)):
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
    dw = jnp.log(gw / pw)
    dh = jnp.log(gh / ph)
    deltas = jnp.stack([dx, dy, dw, dh], axis=-1)

    #means = deltas.new_tensor(means).unsqueeze(0)
    #stds = deltas.new_tensor(stds).unsqueeze(0)
    #deltas = deltas.sub_(means).div_(stds)

    means = jnp.array(means)
    means = jnp.expand_dims(means, 0)
    stds = jnp.array(stds)
    stds = jnp.expand_dims(stds, 0)
    deltas = deltas - means
    deltas = deltas / stds
    return deltas


def args_adaptor(np_args):
    proposals = device_put(np_args[0])
    gt = device_put(np_args[1])

    return [proposals, gt, np_args[2], np_args[3]]


def executer_creator():
    return Executer(legacy_bbox2delta, args_adaptor)
