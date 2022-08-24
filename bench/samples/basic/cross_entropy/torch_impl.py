from functools import reduce
import torch
from torch.nn import functional
from bench.core.executer import Executer


def cross_entropy(input, target, weight_,
            size_average_, ignore_index_, reduce_, reduction_, label_smoothing_):
    output = functional.cross_entropy(input, target, weight=weight_,
            size_average=size_average_, ignore_index=ignore_index_, 
            reduce=reduce_, reduction=reduction_, label_smoothing=label_smoothing_)
    return output


def args_adaptor(np_args):
    input = torch.from_numpy(np_args[0]).cuda()
    target = torch.from_numpy(np_args[1]).cuda()
    weight_ = None
    size_average_ = None
    ignore_index_ = -100
    reduce_ = None
    reduction_ = 'mean'
    label_smoothing_ = 0.0
    if np_args[2] != "":
        if np_args[2] == 'null':
            weight_ = None
        elif isinstance(np_args[2], list):
            weight_ = torch.tensor(np_args[2][0], device=torch.device(np_args[2][1]))
    if np_args[3] != "":
        size_average_ = np_args[3]
    if np_args[4] != "":
        ignore_index_ = np_args[4]
    if np_args[5] != "":
        reduce_ = np_args[5]
    if np_args[6] != "":
        reduction_ = np_args[6]
    if np_args[7] != "":
        label_smoothing_ = np_args[7]
 
    return [input, target, weight_,
            size_average_, ignore_index_, reduce_, reduction_, label_smoothing_]


def executer_creator():
    return Executer(cross_entropy, args_adaptor)
