# Copyright (c) OpenComputeLab. All Rights Reserved.

import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

import torch.nn as nn
import torch.nn.init as init

import onnx

def meshgrid(x, y, row_major=True):
    xx = x.repeat(len(y))
    yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
    if row_major:
        return xx, yy
    else:
        return yy, xx


def valid_flags(featmap_size=(160, 120), valid_size=(160, 120), num_base_anchors=3, device="cuda"):
    feat_h, feat_w = featmap_size
    valid_h, valid_w = valid_size
    assert valid_h <= feat_h and valid_w <= feat_w
    valid_x = torch.zeros(feat_w, dtype=torch.uint8, device=device)
    valid_y = torch.zeros(feat_h, dtype=torch.uint8, device=device)
    valid_x[:valid_w] = 1
    valid_y[:valid_h] = 1
    valid_xx, valid_yy = meshgrid(valid_x, valid_y)
    valid = valid_xx & valid_yy
    valid = (valid[:, None].expand(valid.size(0),
                                   num_base_anchors).contiguous().view(-1))
    return valid


class Bbox(nn.Module):
    def __init__(self):
        super(Bbox, self).__init__()
        self.meshgrid = meshgrid
        self.valid_flags = valid_flags

    def forward(self):
       return self.valid_flags()

torch_model = Bbox()

torch_model.eval()

torch_out = torch_model()

torch.onnx.export(torch_model, 
        (),
        "valid_flags.onnx",
        verbose=True,
        export_params=True,
        opset_version=12,
        input_names=[],
        output_names = ['output'])
