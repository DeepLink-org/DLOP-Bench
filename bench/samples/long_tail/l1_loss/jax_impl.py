# Copyright (c) OpenComputeLab. All Rights Reserved.
import jax.numpy as jnp
import numpy as np
from jax import device_put, grad, value_and_grad
from bench.core.executer import Executer

def l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    #assert pred.size() == target.size() and target.numel() > 0
    loss = jnp.abs(pred - target)
    return loss


def args_adaptor(np_args):
    pred = device_put(np_args[0])
    target = device_put(np_args[1])

    return [pred, target]


def executer_creator():
    return Executer(l1_loss, args_adaptor)
