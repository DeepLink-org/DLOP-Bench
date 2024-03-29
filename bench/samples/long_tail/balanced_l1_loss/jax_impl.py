# Copyright (c) OpenComputeLab. All Rights Reserved.

import jax.numpy as jnp
from jax import device_put
import numpy as np
from bench.core.executer import Executer


def balanced_l1_loss(pred,
                     target,
                     beta=1.0,
                     alpha=0.5,
                     gamma=1.5,
                     reduction="mean"):
    """Calculate balanced L1 loss.

    Please see the `Libra R-CNN <https://arxiv.org/pdf/1904.02701.pdf>`_

    Args:
        pred (jnp.Tensor): The prediction with shape (N, 4).
        target (jnp.Tensor): The learning target of the prediction with
            shape (N, 4).
        beta (float): The loss is a piecewise function of prediction and target
            and ``beta`` serves as a threshold for the difference between the
            prediction and target. Defaults to 1.0.
        alpha (float): The denominator ``alpha`` in the balanced L1 loss.
            Defaults to 0.5.
        gamma (float): The ``gamma`` in the balanced L1 loss.
            Defaults to 1.5.
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".

    Returns:
        jnp.Tensor: The calculated loss
    """
    # assert beta > 0
    # assert pred.size() == target.size() and target.numel() > 0

    diff = jnp.abs(pred - target)
    b = np.e**(gamma / alpha) - 1
    loss = jnp.where(
        diff < beta,
        alpha / b * (b * diff + 1) * jnp.log(b * diff / beta + 1) -
        alpha * diff,
        gamma * diff + gamma / b - alpha * beta,
    )

    return loss


def args_adaptor(np_args):
    pred = device_put(np_args[0])
    target = device_put(np_args[1])
    return [pred, target]


def executer_creator():
    return Executer(balanced_l1_loss, args_adaptor)
