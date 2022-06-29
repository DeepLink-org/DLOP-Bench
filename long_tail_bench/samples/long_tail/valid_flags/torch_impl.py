import torch
from long_tail_bench.core.executer import Executer


def meshgrid(x, y, row_major=True):
    xx = x.repeat(len(y))
    yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
    if row_major:
        return xx, yy
    else:
        return yy, xx


def valid_flags(featmap_size, valid_size, num_base_anchors, device="cuda"):
    feat_h, feat_w = featmap_size
    valid_h, valid_w = valid_size
    assert valid_h <= feat_h and valid_w <= feat_w
    valid_x = torch.zeros(feat_w, dtype=torch.uint8, device=device)
    valid_y = torch.zeros(feat_h, dtype=torch.uint8, device=device)
    valid_x[:valid_w] = 1
    valid_y[:valid_h] = 1
    valid_xx, valid_yy = meshgrid(valid_x, valid_y)
    valid = valid_xx & valid_yy
    valid = (valid[:, None].expand(valid.size(0),
                                   num_base_anchors).contiguous().view(-1))
    return valid


def args_adaptor(np_args):
    featmap_size = tuple(np_args[0].tolist())
    valid_size = tuple(np_args[1].tolist())
    num_base_anchors = np_args[2]

    return [featmap_size, valid_size, num_base_anchors]


def executer_creator():
    return Executer(valid_flags, args_adaptor)
