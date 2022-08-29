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

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


class Bbox(nn.Module):
    def __init__(self):
        super(Bbox, self).__init__()
        self.box_xyxy_to_cxcywh = box_xyxy_to_cxcywh

    def forward(self, x):
        return self.box_xyxy_to_cxcywh(x)


torch_model = Bbox()

torch_model.eval()

x = args_adaptor(gen_np_args(50, 4))
torch_out = torch_model(x)

torch.onnx.export(torch_model, 
        (x),
        "box_xyxy_to_cxcywh.onnx",
        verbose=True,
        export_params=True,
        opset_version=12,
        input_names=['x'],
        output_names = ['output'])
