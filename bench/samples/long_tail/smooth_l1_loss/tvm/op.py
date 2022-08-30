# Copyright (c) OpenComputeLab. All Rights Reserved.

import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

import torch.nn as nn
import torch.nn.init as init
from gen_data import gen_np_args, args_adaptor

import onnx


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


class Bbox(nn.Module):
    def __init__(self):
        super(Bbox, self).__init__()
        self.smooth_l1_loss = smooth_l1_loss

    def forward(self, pred, target):
        loss = self.smooth_l1_loss(pred, target)

        return loss

torch_model = Bbox()

torch_model.eval()

pred, target = args_adaptor(gen_np_args(3000, 4))
torch_out = torch_model(pred, target)

torch.onnx.export(torch_model, 
        (pred, target),
        "smooth_l1_loss.onnx",
        verbose=True,
        export_params=True,
        opset_version=12,
        input_names=['pred', 'target'],
        output_names = ['output'])
