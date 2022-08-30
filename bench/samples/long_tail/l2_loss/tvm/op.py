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
# import onnxoptimizer

def l2_loss(input, target):
    return torch.mean((input - target) * (input - target))


class Bbox(nn.Module):
    def __init__(self):
        super(Bbox, self).__init__()
        self.l2_loss = l2_loss

    def forward(self, inputs, target):
        loss = self.l2_loss(inputs, target)

        return loss 

torch_model = Bbox()

torch_model.eval()

inputs, target = args_adaptor(gen_np_args(16, 32))
torch_out = torch_model(inputs, target)

torch.onnx.export(torch_model, 
        (inputs, target),
        "l2_loss.onnx",
        verbose=True,
        export_params=True,
        opset_version=12,
        input_names=['inputs', 'target'],
        output_names = ['output'])
