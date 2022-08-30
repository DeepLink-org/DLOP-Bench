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

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


class Bbox(nn.Module):
    def __init__(self):
        super(Bbox, self).__init__()
        self.box_cxcywh_to_xyxy = box_cxcywh_to_xyxy

    def forward(self, x):
        ret = self.box_cxcywh_to_xyxy(x)

        return ret

torch_model = Bbox()

torch_model.eval()

x = args_adaptor(gen_np_args(50, 4))
torch_out = torch_model(x)

torch.onnx.export(torch_model, 
        (x),
        "box_cxcywh_to_xyxy.onnx",
        verbose=True,
        export_params=True,
        opset_version=12,
        input_names=['x'],
        output_names = ['output'])
