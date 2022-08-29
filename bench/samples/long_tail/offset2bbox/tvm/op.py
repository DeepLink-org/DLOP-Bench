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
from gen_data import gen_np_args, args_adaptor

import onnx


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


def offset2bbox(boxes, offset, weights=(1.0, 1.0, 1.0, 1.0)):
    ctr_x, ctr_y, widths, heights = xyxy2xywh(boxes)

    wx, wy, ww, wh = weights
    dx = offset[:, 0::4] / wx
    dy = offset[:, 1::4] / wy
    dw = offset[:, 2::4] / ww
    dh = offset[:, 3::4] / wh

    # Prevent sending too large values into np.exp()
    dw = torch.clamp(dw, max=np.log(1000. / 16.))
    dh = torch.clamp(dh, max=np.log(1000. / 16.))

    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = torch.exp(dw) * widths[:, None]
    pred_h = torch.exp(dh) * heights[:, None]

    pred_boxes = offset.new_zeros(offset.shape)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

    return pred_boxes


class Bbox(nn.Module):
    def __init__(self):
        super(Bbox, self).__init__()
        self.xyxy2xywh = xyxy2xywh
        self.offset2bbox = offset2bbox

    def forward(self, x, y):
        z = self.offset2bbox(x, y)
        return z


# Create the super-resolution model by using the above model definition.
torch_model = Bbox()

# set the model to inference mode
torch_model.eval()

# Input to the model
x, y = args_adaptor(gen_np_args(4000, 4))
torch_out = torch_model(x, y)

# Export the model
torch.onnx.export(
    torch_model,  # model being run
    # model input (or a tuple for multiple inputs)
    (x, y),
    # where to save the model (can be a file or file-like object)
    "offset2bbox.onnx",
    verbose=True,
    export_params=
    True,  # store the trained parameter weights inside the model file
    opset_version=12,  # the ONNX version to export the model to
    do_constant_folding=
    True,  # whether to execute constant folding for optimization
    input_names=['input', 'gt'],  # the model's input names
    output_names=['output'],  # the model's output names
)
