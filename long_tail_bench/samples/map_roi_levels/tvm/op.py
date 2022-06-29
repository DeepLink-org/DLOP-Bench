import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

import torch.nn as nn
import torch.nn.init as init
from gen_data import gen_np_args, args_adaptor

import onnx

def map_roi_levels(rois, num_levels=4, finest_scale=56):
    """Map rois to corresponding feature levels by scales.

    - scale < finest_scale * 2: level 0
    - finest_scale * 2 <= scale < finest_scale * 4: level 1
    - finest_scale * 4 <= scale < finest_scale * 8: level 2
    - scale >= finest_scale * 8: level 3

    Args:
        rois (Tensor): Input RoIs, shape (k, 5).
        num_levels (int): Total level number.

    Returns:
        Tensor: Level index (0-based) of each RoI, shape (k, )
    """
    scale = torch.sqrt((rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
    target_lvls = torch.floor(torch.log2(scale / finest_scale + 1e-6))
    target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
    return target_lvls


class Bbox(nn.Module):
    def __init__(self):
        super(Bbox, self).__init__()
        self.map_roi_levels = map_roi_levels

    def forward(self, rois):
        target_lvs = self.map_roi_levels(rois)

        return target_lvs

torch_model = Bbox()

torch_model.eval()

rois = args_adaptor(gen_np_args(20, 5))
torch_out = torch_model(rois)

torch.onnx.export(torch_model, 
        (rois),
        "map_roi_levels.onnx",
        verbose=True,
        export_params=True,
        opset_version=12,
        input_names=['rois'],
        output_names = ['output'])
