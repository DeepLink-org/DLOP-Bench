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
    # pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
    # neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    pos_loss = -(pred + eps).log() * (1 -
                                      pred).pow(alpha) * pos_weights.float()
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights.float()
    return pos_loss + neg_loss


class Bbox(nn.Module):
    def __init__(self):
        super(Bbox, self).__init__()
        self.gaussian_focal_loss = gaussian_focal_loss

    def forward(self, pred, target):
        return self.gaussian_focal_loss(pred, target)

torch_model = Bbox()

torch_model.eval()

pred, target = args_adaptor(gen_np_args(128, 4))
torch_out = torch_model(pred, target)

torch.onnx.export(torch_model, 
        (pred, target),
        "gaussian_focal_loss.onnx",
        verbose=True,
        export_params=True,
        opset_version=12,
        input_names=['pred', 'target'],
        output_names = ['output'])
