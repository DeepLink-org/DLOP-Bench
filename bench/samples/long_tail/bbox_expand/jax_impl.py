# Copyright (c) OpenComputeLab. All Rights Reserved.
import jax.numpy as jnp
import jax.numpy as device_put
from bench.core.executer import Executer


def bbox_target_expand(bbox_targets, bbox_weights, labels, bbox_targets_expand, bbox_weights_expand):
    """jax.core.ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected.
    The error arose in jnp.nonzero"""
    # Obviously caused by value-dependency
    for i in jnp.array(jnp.nonzero(labels > 0)).squeeze(0):
        start, end = labels[i] * 4, (labels[i] + 1) * 4
        """TypeError: '<class 'jax.interpreters.xla.DeviceArray'>' object does not support item assignment. 
        JAX arrays are immutable; perhaps you want jax.ops.index_update or jax.ops.index_add instead?"""
        bbox_targets_expand[i, start:end] = bbox_targets[i, :]
        bbox_weights_expand[i, start:end] = bbox_weights[i, :]
    return bbox_targets_expand, bbox_weights_expand


def args_adaptor(np_args):
    bbox_targets = device_put(np_args[0])
    bbox_weights = device_put(np_args[1])
    labels = device_put(np_args[2])
    bbox_targets_expand = device_put(np_args[3])
    bbox_weights_expand = device_put(np_args[4])

    return [
        bbox_targets,
        bbox_weights,
        labels,
        bbox_targets_expand,
        bbox_weights_expand,
    ]


def executer_creator():
    return Executer(bbox_target_expand, args_adaptor)
