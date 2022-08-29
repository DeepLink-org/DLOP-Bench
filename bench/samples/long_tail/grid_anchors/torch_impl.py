import torch
from long_tail_bench.core.executer import Executer


def meshgrid(x, y, row_major=True):
    xx = x.repeat(len(y))
    yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
    if row_major:
        return xx, yy
    else:
        return yy, xx


def grid_anchors(base_anchors, featmap_size, stride, device="cuda"):

    feat_h, feat_w = featmap_size
    shift_x = torch.arange(0, feat_w, device=device) * stride
    shift_y = torch.arange(0, feat_h, device=device) * stride
    shift_xx, shift_yy = meshgrid(shift_x, shift_y)
    shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
    shifts = shifts.type_as(base_anchors)

    all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
    all_anchors = all_anchors.view(-1, 4)
    return all_anchors


def args_adaptor(np_args):
    base_anchors = torch.from_numpy(np_args[0]).cuda()

    return [base_anchors, np_args[1], np_args[2]]


def executer_creator():
    return Executer(grid_anchors, args_adaptor)
