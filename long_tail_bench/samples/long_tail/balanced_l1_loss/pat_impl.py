import torch
import numpy as np
from long_tail_bench.core.executer import Executer


def balanced_l1_loss(pred,
                     target,
                     beta=1.0,
                     alpha=0.5,
                     gamma=1.5,
                     reduction="mean"):
    """Calculate balanced L1 loss.

    Please see the `Libra R-CNN <https://arxiv.org/pdf/1904.02701.pdf>`_

    Args:
        pred (torch.Tensor): The prediction with shape (N, 4).
        target (torch.Tensor): The learning target of the prediction with
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
        torch.Tensor: The calculated loss
    """
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0

    diff = torch.abs(pred - target)
    b = np.e**(gamma / alpha) - 1
    loss = torch.where(
        diff < beta,
        alpha / b * (b * diff + 1) * torch.log(b * diff / beta + 1) -
        alpha * diff,
        gamma * diff + gamma / b - alpha * beta,
    )

    return loss


def args_adaptor(np_args):
    pred = torch.from_numpy(np_args[0]).cuda()
    pred.requires_grad = True
    target = torch.from_numpy(np_args[1]).cuda()

    return [pred, target]


def executer_creator():
    return Executer(balanced_l1_loss, args_adaptor)
