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

def legacy_bbox2delta(proposals,
                      gt,
                      means=(0.0, 0.0, 0.0, 0.0),
                      stds=(1.0, 1.0, 1.0, 1.0)):
    """Compute deltas of proposals w.r.t. gt in the MMDet V1.x manner.

    We usually compute the deltas of x, y, w, h of proposals w.r.t ground
    truth bboxes to get regression target.
    This is the inverse function of `delta2bbox()`

    Args:
        proposals (Tensor): Boxes to be transformed, shape (N, ..., 4)
        gt (Tensor): Gt bboxes to be used as base, shape (N, ..., 4)
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates

    Returns:
        Tensor: deltas with shape (N, 4), where columns represent dx, dy,
            dw, dh.
    """
    assert proposals.size() == gt.size()

    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0] + 1.0
    ph = proposals[..., 3] - proposals[..., 1] + 1.0

    gx = (gt[..., 0] + gt[..., 2]) * 0.5
    gy = (gt[..., 1] + gt[..., 3]) * 0.5
    gw = gt[..., 2] - gt[..., 0] + 1.0
    gh = gt[..., 3] - gt[..., 1] + 1.0

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    deltas = torch.stack([dx, dy, dw, dh], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


class Bbox(nn.Module):
    def __init__(self):
        super(Bbox, self).__init__()
        self.legacy_bbox2delta = legacy_bbox2delta

    def forward(self, proposals, gt):
        deltas = self.legacy_bbox2delta(proposals, gt)

        return deltas

torch_model = Bbox()

torch_model.eval()

proposals, gt = args_adaptor(gen_np_args(3000))
torch_out = torch_model(proposals, gt)

torch.onnx.export(torch_model, 
        (proposals, gt),
        "legacy_bbox2delta.onnx",
        verbose=True,
        export_params=True,
        opset_version=12,
        input_names=['porposals', 'gt'],
        output_names = ['output'])
