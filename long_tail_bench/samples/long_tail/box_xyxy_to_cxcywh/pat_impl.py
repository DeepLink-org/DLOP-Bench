import torch
from long_tail_bench.core.executer import Executer


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def args_adaptor(np_args):
    x = torch.from_numpy(np_args[0]).cuda()
    return [x]


def executer_creator():
    return Executer(box_xyxy_to_cxcywh, args_adaptor)
