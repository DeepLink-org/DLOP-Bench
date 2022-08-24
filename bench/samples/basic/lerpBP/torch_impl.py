import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def lerp(input_torch, end_torch, weight):
    ret = torch.lerp(input_torch, end_torch, weight)
    ret.backward(ret)
    return ret


def args_adaptor(np_args):
    input_torch = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    end_torch = torch.from_numpy(np_args[1]).to(torch.float32).cuda()
    input_torch.requires_grad = True
    end_torch.requires_grad = True

    return [input_torch, end_torch, np_args[2]]


def executer_creator():
    return Executer(lerp, args_adaptor)
