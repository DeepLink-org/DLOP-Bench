import torch
import torch.nn.functional as F
from long_tail_bench.core.executer import Executer


def distribution_focal_loss(pred, label):
    r"""Distribution Focal Loss (DFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

    Args:
        pred (torch.Tensor): Predicted general distribution of bounding boxes
            (before softmax) with shape (N, n+1), n is the max value of the
            integral set `{0, ..., n}` in paper.
        label (torch.Tensor): Target distance label for bounding boxes with
            shape (N,).

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    dis_left = label.long()
    dis_right = dis_left + 1
    weight_left = dis_right.float() - label
    weight_right = label - dis_left.float()
    loss = (F.cross_entropy(pred, dis_left, reduction="none") * weight_left +
            F.cross_entropy(pred, dis_right, reduction="none") * weight_right)
    return loss


def args_adaptor(np_args):
    pred = torch.from_numpy(np_args[0]).cuda()
    pred.requires_grad = True
    target = torch.from_numpy(np_args[1]).cuda()
    return [pred, target]


def executer_creator():
    return Executer(distribution_focal_loss, args_adaptor)
