import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch.onnx

import torch.nn as nn
import torch.nn.init as init
from gen_data import gen_np_args, args_adaptor

import onnx


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


class Bbox(nn.Module):
    def __init__(self):
        super(Bbox, self).__init__()
        self.distribution_focal_loss = distribution_focal_loss

    def forward(self, pred, label):
        loss = self.distribution_focal_loss(pred, label)

        return loss

torch_model = Bbox()

torch_model.eval()

pred, target = args_adaptor(gen_np_args(128, 4))
torch_out = torch_model(pred, target)

torch.onnx.export(torch_model, 
        (pred, target),
        "distribution_focal_loss.onnx",
        verbose=True,
        export_params=True,
        opset_version=12,
        input_names=['pred', 'target'],
        output_names = ['output'])
