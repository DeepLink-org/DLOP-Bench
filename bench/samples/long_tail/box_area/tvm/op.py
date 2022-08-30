# Copyright (c) OpenComputeLab. All Rights Reserved.

# Some standard imports
import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

import torch.nn as nn
import torch.nn.init as init
from gen_data import gen_np_args, args_adaptor

import onnx


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


class model(nn.Module):
    def __init__(self, upscale_factor, inplace=False):
        super(model, self).__init__()
        self.graph = box_area

    def forward(self, x):
        return self.graph(x)


torch_model = model(upscale_factor=3)
torch_model.eval()

x = args_adaptor(gen_np_args(20, 5))
torch_out = torch_model(x)

# Export the model
torch.onnx.export(
    torch_model,  # model being run
    # model input (or a tuple for multiple inputs)
    (x),
    # where to save the model (can be a file or file-like object)
    "box_area.onnx",
    verbose=False,
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=12,  # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    #input_names=['input_1', 'input_2'],  # the model's input names
    #output_names=['output_1', 'output_2'],  # the model's output names
)

model = onnx.load('box_area.onnx')
print('Model:\n\n{}'.format(onnx.helper.printable_graph(model.graph)))
for node in model.graph.initializer:
    print(node)
