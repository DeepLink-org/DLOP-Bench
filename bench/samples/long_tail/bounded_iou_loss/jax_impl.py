# Copyright (c) OpenComputeLab. All Rights Reserved.

import jax.numpy as jnp
from jax import device_put
import numpy as np
from bench.core.executer import Executer


def bounded_iou_loss(pred, target, beta=0.2, eps=1e-3):
    """BIoULoss.

    This is an implementation of paper
    `Improving Object Localization with Fitness NMS and Bounded IoU Loss.
    <https://arxiv.org/abs/1711.00164>`_.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Target bboxes.
        beta (float): beta parameter in smoothl1.
        eps (float): eps to avoid NaN.
    """
    pred_ctrx = (pred[:, 0] + pred[:, 2]) * 0.5
    pred_ctry = (pred[:, 1] + pred[:, 3]) * 0.5
    pred_w = pred[:, 2] - pred[:, 0]
    pred_h = pred[:, 3] - pred[:, 1]
    target_ctrx = (target[:, 0] + target[:, 2]) * 0.5
    target_ctry = (target[:, 1] + target[:, 3]) * 0.5
    target_w = target[:, 2] - target[:, 0]
    target_h = target[:, 3] - target[:, 1]

    dx = target_ctrx - pred_ctrx
    dy = target_ctry - pred_ctry

    loss_dx = 1 - jnp.maximum(
        (target_w - 2 * jnp.abs(dx)) /
        (target_w + 2 * jnp.abs(dx) + eps), jnp.zeros(dx.shape))
    loss_dy = 1 - jnp.maximum(
        (target_h - 2 * jnp.abs(dy)) /
        (target_h + 2 * jnp.abs(dy) + eps), jnp.zeros(dy.shape))
    loss_dw = 1 - jnp.minimum(target_w / (pred_w + eps), pred_w /
                            (target_w + eps))
    loss_dh = 1 - jnp.minimum(target_h / (pred_h + eps), pred_h /
                            (target_h + eps))
    loss_comb = jnp.reshape(jnp.stack([loss_dx, loss_dy, loss_dw, loss_dh],
                            axis=-1), [loss_dx.shape[0], -1])

    loss = jnp.where(loss_comb < beta, 0.5 * loss_comb * loss_comb / beta,
                       loss_comb - 0.5 * beta)

    return loss


def args_adaptor(np_args):
    pred = device_put(np_args[0])
    target = device_put(np_args[1])
    return [pred, target]


def executer_creator():
    return Executer(bounded_iou_loss, args_adaptor)
