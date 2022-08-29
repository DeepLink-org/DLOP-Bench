# Copyright (c) OpenComputeLab. All Rights Reserved.
import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from gen_data import gen_np_args, args_adaptor

import onnx

def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction="mean", avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == "mean":
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != "none":
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


class Bbox(nn.Module):
    def __init__(self):
        super(Bbox, self).__init__()
        self.weight_reduce_loss = weight_reduce_loss

    def forward(self, loss_in, weight_in):
        loss = self.weight_reduce_loss(loss_in, weight_in)

        return loss 

torch_model = Bbox()

torch_model.eval()

loss_in, weight_in = args_adaptor(gen_np_args(128, 4))
torch_out = torch_model(loss_in, weight_in)

torch.onnx.export(torch_model, 
        (loss_in, weight_in),
        "weight_reduce_loss.onnx",
        verbose=True,
        export_params=True,
        opset_version=12,
        input_names=['loss_in', 'weight_in'],
        output_names = ['output'])
