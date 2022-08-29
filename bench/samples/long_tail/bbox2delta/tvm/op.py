import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

import torch.nn as nn
import torch.nn.init as init

import onnx
from gen_data import gen_np_args, args_adaptor

def bbox2delta(proposals,
               gt,
               means=(0.0, 0.0, 0.0, 0.0),
               stds=(1.0, 1.0, 1.0, 1.0)):
    assert proposals.size() == gt.size()

    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]

    gx = (gt[..., 0] + gt[..., 2]) * 0.5
    gy = (gt[..., 1] + gt[..., 3]) * 0.5
    gw = gt[..., 2] - gt[..., 0]
    gh = gt[..., 3] - gt[..., 1]

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
        self.bbox2delta = bbox2delta

    def forward(self, proposals, gt):
        return self.bbox2delta(proposals, gt)

torch_model = Bbox()
torch_model.eval()

proposals, gt = args_adaptor(gen_np_args(128))
torch_out = torch_model(proposals, gt)

torch.onnx.export(torch_model,
        (proposals, gt),
        "bbox2delta.onnx",
        verbose=True,
        export_params=True,
        opset_version=12,
        input_names=['porposals', 'gt'],
        output_names = ['output'])
