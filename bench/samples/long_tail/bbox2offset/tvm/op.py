# Copyright (c) OpenComputeLab. All Rights Reserved.

# Some standard imports
import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

# Super Resolution model definition in PyTorch
import torch.nn as nn
import torch.nn.init as init

import onnx
from gen_data import gen_np_args, args_adaptor


def xyxy2xywh(boxes, stacked=False):
    """(x1, y1, x2, y2) -> (x, y, w, h)"""
    x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    w = (x2 - x1 + 1)
    h = (y2 - y1 + 1)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    if stacked:
        return torch.stack([cx, cy, w, h], dim=1)
    else:
        return cx, cy, w, h


def bbox2offset(boxes, gt_boxes, weights=(1.0, 1.0, 1.0, 1.0)):
    # assert boxes.shape[0] == gt_boxes.shape[0]
    ex_ctr_x, ex_ctr_y, ex_widths, ex_heights = xyxy2xywh(boxes, False)
    gt_ctr_x, gt_ctr_y, gt_widths, gt_heights = xyxy2xywh(gt_boxes, False)

    wx, wy, ww, wh = weights
    offset_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    offset_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    offset_dw = ww * torch.log(gt_widths / ex_widths)
    offset_dh = wh * torch.log(gt_heights / ex_heights)
    offset = torch.stack((offset_dx, offset_dy, offset_dw, offset_dh), dim=1)
    return offset


class Bbox(nn.Module):
    def __init__(self):
        super(Bbox, self).__init__()
        self.xyxy2xywh = xyxy2xywh
        self.bbox2offset = bbox2offset

    def forward(self, x, y):
        z = self.bbox2offset(x, y)
        return z


# Create the super-resolution model by using the above model definition.
torch_model = Bbox()

# set the model to inference mode
torch_model.eval()

# Input to the model
x, y = args_adaptor(gen_np_args(3000, 4))
torch_out = torch_model(x, y)

# Export the model
torch.onnx.export(
    torch_model,  # model being run
    # model input (or a tuple for multiple inputs)
    (x, y),
    # where to save the model (can be a file or file-like object)
    "bbox2offset.onnx",
    verbose=True,
    export_params=
    True,  # store the trained parameter weights inside the model file
    opset_version=12,  # the ONNX version to export the model to
    do_constant_folding=
    True,  # whether to execute constant folding for optimization
    input_names=['input', 'gt'],  # the model's input names
    output_names=['output'],  # the model's output names
)

# model = onnx.load('bbox2offset.onnx')
# sim_model = onnxoptimizer.optimize(model)
# # print('Model:\n\n{}'.format(onnx.helper.printable_graph(model.graph)))
# for node in model.graph.node:
#     print(node)
# model = onnx.load('bbox2offset_sim.onnx')
# print('Model:\n\n{}'.format(onnx.helper.printable_graph(model.graph)))
