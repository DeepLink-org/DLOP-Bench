# Copyright(c) OpenMMLab. All Rights Reserved.
# Copied from
import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

import torch.nn as nn
import torch.nn.init as init
from gen_data import gen_np_args, args_adaptor

import onnx


def l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert pred.size() == target.size() and target.numel() > 0
    loss = torch.abs(pred - target)
    return loss


class Bbox(nn.Module):
    def __init__(self):
        super(Bbox, self).__init__()
        self.l1_loss = l1_loss

    def forward(self, pred, target):
        loss = self.l1_loss(pred, target)

        return loss 

torch_model = Bbox()

torch_model.eval()

pred, target = args_adaptor(gen_np_args(128, 4))
torch_out = torch_model(pred, target)

torch.onnx.export(torch_model, 
        (pred, target),
        "l1_loss.onnx",
        verbose=True,
        export_params=True,
        opset_version=12,
        input_names=['pred', 'target'],
        output_names = ['output'])
