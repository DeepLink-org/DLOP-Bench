import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

import torch.nn as nn
import torch.nn.init as init
from gen_data import gen_np_args, args_adaptor

import onnx


def yolo_decode(bboxes, pred_bboxes, stride):
    """Apply transformation `pred_bboxes` to `boxes`.

    Args:
        boxes (torch.Tensor): Basic boxes, e.g. anchors.
        pred_bboxes (torch.Tensor): Encoded boxes with shape
        stride (torch.Tensor | int): Strides of bboxes.

    Returns:
        torch.Tensor: Decoded boxes.
    """
    assert pred_bboxes.size(0) == bboxes.size(0)
    assert pred_bboxes.size(-1) == bboxes.size(-1) == 4
    x_center = (bboxes[..., 0] + bboxes[..., 2]) * 0.5
    y_center = (bboxes[..., 1] + bboxes[..., 3]) * 0.5
    w = bboxes[..., 2] - bboxes[..., 0]
    h = bboxes[..., 3] - bboxes[..., 1]
    # Get outputs x, y
    x_center_pred = (pred_bboxes[..., 0] - 0.5) * stride + x_center
    y_center_pred = (pred_bboxes[..., 1] - 0.5) * stride + y_center
    w_pred = torch.exp(pred_bboxes[..., 2]) * w
    h_pred = torch.exp(pred_bboxes[..., 3]) * h

    decoded_bboxes = torch.stack(
        (
            x_center_pred - w_pred / 2,
            y_center_pred - h_pred / 2,
            x_center_pred + w_pred / 2,
            y_center_pred + h_pred / 2,
        ),
        dim=-1,
    )

    return decoded_bboxes


class Bbox(nn.Module):
    def __init__(self):
        super(Bbox, self).__init__()
        self.yolo_decode = yolo_decode

    def forward(self, bboxes, pred_bboxes, stride):
        decoded_bboxes = self.yolo_decode(bboxes, pred_bboxes, stride)

        return decoded_bboxes

torch_model = Bbox()

torch_model.eval()

bboxes, pred_bboxes, stride = args_adaptor(gen_np_args(3000, 4))
torch_out = torch_model(bboxes, pred_bboxes, stride)

torch.onnx.export(torch_model, 
        (bboxes, pred_bboxes, stride),
        "yolo_decode.onnx",
        verbose=True,
        export_params=True,
        opset_version=12,
        input_names=['bboxes', 'pred_bboxes', 'stride'],
        output_names = ['output'])
