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

def delta2bbox(
    rois,
    deltas,
    means=[0, 0, 0, 0],
    stds=[1, 1, 1, 1],
    max_shape=None,
    wh_ratio_clip=16 / 1000,
):
    if isinstance(deltas, torch.cuda.HalfTensor):
        deltas = deltas.float()
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 4)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 4)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::4]
    dy = denorm_deltas[:, 1::4]
    dw = denorm_deltas[:, 2::4]
    dh = denorm_deltas[:, 3::4]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
    pw = (rois[:, 2] - rois[:, 0] + 1.0).unsqueeze(1).expand_as(dw)
    ph = (rois[:, 3] - rois[:, 1] + 1.0).unsqueeze(1).expand_as(dh)
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    tmp = pw * dx
    gx = px + tmp
    tmp = ph * dy
    gy = py + tmp
    # gx = torch.addcmul(px, 1, pw, dx)  # gx = px + pw * dx
    # gy = torch.addcmul(py, 1, ph, dy)  # gy = py + ph * dy
    x1 = gx - gw * 0.5 + 0.5
    y1 = gy - gh * 0.5 + 0.5
    x2 = gx + gw * 0.5 - 0.5
    y2 = gy + gh * 0.5 - 0.5
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1] - 1)
        y1 = y1.clamp(min=0, max=max_shape[0] - 1)
        x2 = x2.clamp(min=0, max=max_shape[1] - 1)
        y2 = y2.clamp(min=0, max=max_shape[0] - 1)
    bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view_as(deltas)
    return bboxes


class Bbox(nn.Module):
    def __init__(self):
        super(Bbox, self).__init__()
        self.delta2bbox = delta2bbox

    def forward(self, rois, deltas):
        bboxes = self.delta2bbox(rois, deltas)

        return bboxes

torch_model = Bbox()

torch_model.eval()

rois, deltas = args_adaptor(gen_np_args(3000, 4))
torch_out = torch_model(rois, deltas)

torch.onnx.export(torch_model,
        (rois, deltas),
        "delta2bbox.onnx",
        verbose=True,
        export_params=True,
        opset_version=12,
        input_names=['rois', 'deltas'],
        output_names = ['output'])
