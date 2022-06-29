import torch
from long_tail_bench.core.executer import Executer


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def args_adaptor(np_args):
    boxes = torch.from_numpy(np_args[0]).cuda()
    return [boxes]


def executer_creator():
    return Executer(box_area, args_adaptor)
