# Copyright (c) OpenComputeLab. All Rights Reserved.
import jax
import jax.numpy as jnp
from jax import device_put
from bench.core.executer import Executer


def label_aware_attention(keys, query):
    """label_aware_attention
    """
    weight = jnp.sum(keys * query, axis=-1)
    # weight = jax.lax.pow(weight, 2)  # [x,k_max,1]
    weight = weight * weight
    weight = jax.nn.softmax(weight, 0)
    output = jnp.sum(keys * weight, axis=1)
    return output, weight


def args_adaptor(np_args):
    keys = device_put(np_args[0])
    query = device_put(np_args[1])
    return [keys, query]


def executer_creator():
    return Executer(label_aware_attention, args_adaptor)
