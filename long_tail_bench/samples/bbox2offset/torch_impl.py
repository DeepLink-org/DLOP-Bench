import torch
from long_tail_bench.core.executer import Executer


def xyxy2xywh(boxes):
    """(x1, y1, x2, y2) -> (x, y, w, h)"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w = (x2 - x1 + 1)
    h = (y2 - y1 + 1)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return cx, cy, w, h


def bbox2offset(boxes, gt_boxes, weights: tuple):
    assert boxes.shape[0] == gt_boxes.shape[0]
    ex_ctr_x, ex_ctr_y, ex_widths, ex_heights = xyxy2xywh(boxes)
    gt_ctr_x, gt_ctr_y, gt_widths, gt_heights = xyxy2xywh(gt_boxes)

    wx, wy, ww, wh = weights
    offset_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    offset_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    offset_dw = ww * torch.log(gt_widths / ex_widths)
    offset_dh = wh * torch.log(gt_heights / ex_heights)
    offset = torch.stack((offset_dx, offset_dy, offset_dw, offset_dh), dim=1)
    return offset


def args_adaptor(np_args):
    boxes = torch.from_numpy(np_args[0]).cuda()
    gt = torch.from_numpy(np_args[1]).cuda()
    weights = (1.0, 1.0, 1.0, 1.0)

    return [boxes, gt, weights]


def executer_creator():
    return Executer(bbox2offset, args_adaptor)
