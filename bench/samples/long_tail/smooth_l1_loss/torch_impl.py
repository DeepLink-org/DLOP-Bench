import torch
from long_tail_bench.core.executer import Executer


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
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss


def fast_smooth_l1_loss(pred, target):
    """Smooth L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < 1.0, 0.5 * diff * diff / 1.0, diff - 0.5 * 1.0)
    return loss


def args_generator(M, N):
    shape = (M, N)
    pred = torch.randn(shape, device="cuda")
    target = torch.randn(shape, device="cuda")
    pred.requires_grad = True
    return [pred, target]


def executer_creator():
    return Executer(smooth_l1_loss, args_generator)
