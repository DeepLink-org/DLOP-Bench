import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def kl_div(input_torch, target_torch, reduction):
    return torch.nn.functional.kl_div(input_torch, target_torch, reduction=reduction)


def args_adaptor(np_args):
    input_torch = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    target_torch = torch.from_numpy(np_args[1]).to(torch.float32).cuda()
    return [input_torch, target_torch, np_args[2]]


def executer_creator():
    return Executer(kl_div, args_adaptor)
