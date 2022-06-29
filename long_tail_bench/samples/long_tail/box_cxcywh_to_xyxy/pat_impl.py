# from tiny_ops.jit_example.mmdet_accuracy import N
import torch
from long_tail_bench.core.executer import Executer


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def args_adaptor(np_args):
    x = torch.from_numpy(np_args[0]).cuda()
    return [x]


def executer_creator():
    return Executer(box_cxcywh_to_xyxy, args_adaptor)
