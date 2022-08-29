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

def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks,
    (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float, device="cuda")
    x = torch.arange(0, w, dtype=torch.float, device="cuda")
    y, x = torch.meshgrid(y, x)

    x_mask = masks * x.unsqueeze(0)
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = masks * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


class Bbox(nn.Module):
    def __init__(self):
        super(Bbox, self).__init__()
        self.masks_to_boxes = masks_to_boxes

    def forward(self, masks):
        ret = self.masks_to_boxes(masks)

        return ret

torch_model = Bbox()

torch_model.eval()

masks = args_adaptor(gen_np_args(400, 128, 6))
torch_out = torch_model(masks)

torch.onnx.export(torch_model, 
        (masks),
        "masks_to_boxes.onnx",
# import onnxoptimizer
        verbose=True,
        export_params=True,
        opset_version=12,
        input_names=['masks'],
        output_names = ['output'])
