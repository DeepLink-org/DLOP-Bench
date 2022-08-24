import numpy as np
import tensorflow as tf
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
    loss = tf.abs(pred - target)
    return loss


def args_adaptor(np_args):
    pred = tf.convert_to_tensor(np_args[0], tf.float32)
    target = tf.convert_to_tensor(np_args[1], tf.float32)
    # pred.requires_grad = True

    return [pred, target]


def executer_creator():
    return Executer(l1_loss, args_adaptor)
