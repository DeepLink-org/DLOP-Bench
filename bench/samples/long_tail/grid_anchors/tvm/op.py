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


def meshgrid(x, y, row_major=True):
    xx = x.repeat(len(y))
    yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
    if row_major:
        return xx, yy
    else:
        return yy, xx


def grid_anchors(base_anchors, featmap_size=(16, 32), stride=8, device="cuda"):
    feat_h, feat_w = featmap_size
    shift_x = torch.arange(0, feat_w, device=device) * stride
    shift_y = torch.arange(0, feat_h, device=device) * stride
    shift_xx, shift_yy = meshgrid(shift_x, shift_y)
    shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
    shifts = shifts.type_as(base_anchors)
    # first feat_w elements correspond to the first row of shifts
    # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
    # shifted anchors (K, A, 4), reshape to (K*A, 4)

    all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
    all_anchors = all_anchors.view(-1, 4)
    # first A rows correspond to A anchors of (0, 0) in feature map,
    # then (0, 1), (0, 2), ...
    return all_anchors


class Bbox(nn.Module):
    def __init__(self):
        super(Bbox, self).__init__()
        self.meshgrid = meshgrid
        self.grid_anchors = grid_anchors

    def forward(self, base_anchors):
        all_anchors = self.grid_anchors(base_anchors)

        return all_anchors

torch_model = Bbox()

torch_model.eval()

base_anchors = args_adaptor(gen_np_args(16, 32, 8))
torch_out = torch_model(base_anchors)

torch.onnx.export(torch_model, 
        (base_anchors),
        "grid_anchors.onnx",
        verbose=True,
        export_params=True,
        opset_version=12,
        input_names=['base_anchors'],
        output_names = ['output'])
