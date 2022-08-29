import torch
import numpy as np
from long_tail_bench.core.executer import Executer


def allow_empty_tensor(num=1, empty_shape=(0, 4)):
    """
    Return an empty tensor directly if any of first `num` argument is empty
    """
    def decorate(func):
        def wrapper(*args, **kwargs):
            for arg in args[:num]:
                if torch.is_tensor(arg) and arg.numel() == 0:
                    return arg.new_zeros(empty_shape)
            return func(*args, **kwargs)

        return wrapper

    return decorate


@allow_empty_tensor(1)
def xyxy2xywh(boxes, stacked=False):
    """(x1, y1, x2, y2) -> (x, y, w, h)"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    if stacked:
        return torch.stack([cx, cy, w, h], dim=1)
    else:
        return cx, cy, w, h


# TODO(limaolin): coderized func returns different value now.
@allow_empty_tensor(2)
def offset2bbox(boxes, offset, weights=(1.0, 1.0, 1.0, 1.0)):
    """
    Forward transform that maps proposal boxes to predicted ground-truth
    boxes using bounding-box regression deltas(offset). See bbox_transform_inv
    for a description of the weights argument.
    """
    ctr_x, ctr_y, widths, heights = xyxy2xywh(boxes)

    wx, wy, ww, wh = weights
    dx = offset[:, 0::4] / wx
    dy = offset[:, 1::4] / wy
    dw = offset[:, 2::4] / ww
    dh = offset[:, 3::4] / wh

    # Prevent sending too large values into np.exp()
    dw = torch.clamp(dw, max=np.log(1000.0 / 16.0))
    dh = torch.clamp(dh, max=np.log(1000.0 / 16.0))

    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = torch.exp(dw) * widths[:, None]
    pred_h = torch.exp(dh) * heights[:, None]

    # For parrots jit: new_zeros empty op shape is fixed
    # because of python value offset.shape
    pred_boxes = offset.zeros_like(offset)
    # pred_boxes = offset.new_zeros(offset.shape)

    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

    return pred_boxes


def args_adaptor(np_args):
    boxes = torch.from_numpy(np_args[0]).cuda()
    offset = torch.from_numpy(np_args[1]).cuda()
    weights = tuple(np_args[2].tolist())

    return [boxes, offset, weights]


def executer_creator():
    return Executer(offset2bbox, args_adaptor)
