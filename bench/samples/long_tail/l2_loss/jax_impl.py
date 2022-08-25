# Copyright(c) OpenMMLab. All Rights Reserved.
# Copied from
from jax import numpy as np
from jax import device_put
from bench.core.executer import Executer


def l2_loss(input, target):
    return np.mean((input - target) * (input -target))


def args_generator(np_args):
    output = device_put(np_args[0])
    target = device_put(np_args[1])
    return [output, target]


def executer_creator():
    return Executer(l2_loss, args_generator)
