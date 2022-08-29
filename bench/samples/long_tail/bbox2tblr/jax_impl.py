# Copyright (c) OpenComputeLab. All Rights Reserved.
from jax import numpy as np
import jax.numpy as jnp
from jax import jit, device_put
from bench.core.executer import Executer


def bboxes2tblr(priors, gts, normalizer=4.0, normalize_by_wh=True):
    # dist b/t match center and prior's center
    prior_centers = (priors[:, 0:2] + priors[:, 2:4]) / 2
    xmin, ymin, xmax, ymax = jnp.split(gts, 4, axis=1)
    
    top = jnp.expand_dims(prior_centers[:, 1], 1) - ymin
    bottom = ymax - jnp.expand_dims(prior_centers[:, 1], 1)
    left = jnp.expand_dims(prior_centers[:, 0], 1) - xmin
    right = xmax - jnp.expand_dims(prior_centers[:, 0], 1)
    loc = jnp.concatenate((top, bottom, left, right), axis=1)
    if normalize_by_wh:
        # Normalize tblr by anchor width and height
        wh = priors[:, 2:4] - priors[:, 0:2]
        w, h = jnp.split(wh, 2, axis=1)
        loc[:, :2] /= h  # tb is normalized by h
        loc[:, 2:] /= w  # lr is normalized by w
    # Normalize tblr by the given normalization factor
    return loc / normalizer


def args_adaptor(np_args):
    priors = device_put(np_args[0])
    gts = device_put(np_args[1])
    return [priors, gts]


def executer_creator():
    return Executer(bboxes2tblr, args_adaptor)
