# Copyright (c) OpenComputeLab. All Rights Reserved.
import tensorflow as tf
import numpy as np
from bench.core.executer import Executer


def smooth_l1_loss(pred, target, beta=1.0):
    """Smooth L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Calculated loss
    """
    diff = tf.math.abs(pred - target)
    loss = tf.where(diff < beta, 0.5 * diff * diff / beta,
                    diff - 0.5 * beta)
    return loss


def args_adaptor(np_args):
    pred = tf.convert_to_tensor(np_args[0], tf.float32)
    target = tf.convert_to_tensor(np_args[1], tf.float32)
    # pred.requires_grad = True

    return [pred, target]


def executer_creator():
    return Executer(smooth_l1_loss, args_adaptor)
