import torch
from long_tail_bench.core.executer import Executer


def gaussian_focal_loss(pred, gaussian_target, alpha=2.0, gamma=4.0):
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
    distribution.

    Args:
        pred (torch.Tensor): The prediction.
        gaussian_target (torch.Tensor): The learning target of the prediction
            in gaussian distribution.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
    """
    eps = 1e-12
    pos_weights = gaussian_target.eq(1)
    neg_weights = (1 - gaussian_target).pow(gamma)
    # NOTE(lizhouyang): float * bool is not supported by Parrots.
    # pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
    # neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    pos_loss = -(pred + eps).log() * (1 -
                                      pred).pow(alpha) * pos_weights.float()
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights.float()
    return pos_loss + neg_loss


def args_adaptor(np_args):
    pred = torch.from_numpy(np_args[0]).cuda()
    pred.requires_grad = True
    target = torch.from_numpy(np_args[1]).cuda()
    return [pred, target]


def executer_creator():
    return Executer(gaussian_focal_loss, args_adaptor)
