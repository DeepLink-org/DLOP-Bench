# Copyright (c) OpenComputeLab. All Rights Reserved.

# Some standard imports
import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

import torch.nn as nn
import torch.nn.init as init
from gen_data import args_adaptor, gen_np_args

import onnx


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


class model(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(model, self).__init__()
        self.graph = box_iou

    def forward(self, x, y):
        return self.graph(x, y)


torch_model = model(upscale_factor=3)
torch_model.eval()


x, y = args_adaptor(gen_np_args(400, 4, 72, 4))
torch_out = torch_model(x, y)

# Export the model
torch.onnx.export(
    torch_model,  # model being run
    # model input (or a tuple for multiple inputs)
    (x, y),
    # where to save the model (can be a file or file-like object)
    "box_iou.onnx",
    verbose=False,
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=12,  # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    #input_names=['input_1', 'input_2'],  # the model's input names
    #output_names=['output_1', 'output_2'],  # the model's output names
)
