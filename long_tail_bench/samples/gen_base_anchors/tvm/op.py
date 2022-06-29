import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

import torch.nn as nn
import torch.nn.init as init
from gen_data import gen_np_args, args_adaptor

import onnx

def gen_base_anchors(ratios, scales, base_size=4):
    w = base_size
    h = base_size
    x_ctr = 0.5 * (w - 1)
    y_ctr = 0.5 * (h - 1)

    h_ratios = torch.sqrt(ratios)
    w_ratios = 1 / h_ratios
    ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
    hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)

    base_anchors = torch.stack(
        [
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1),
        ],
        dim=-1,
    ).round()

    return base_anchors


class Bbox(nn.Module):
    def __init__(self):
        super(Bbox, self).__init__()
        self.gen_base_anchors = gen_base_anchors

    def forward(self, ratios, scales):
        return self.gen_base_anchors(ratios, scales)

torch_model = Bbox()

torch_model.eval()

ratios, scales = args_adaptor(gen_np_args(4, 3, 1))

torch_out = torch_model(ratios, scales)

torch.onnx.export(torch_model, 
        (ratios, scales),
        "gen_base_anchors.onnx",
        verbose=True,
        export_params=True,
        opset_version=12,
        input_names=['ratios', 'scales'],
        output_names = ['output'])
