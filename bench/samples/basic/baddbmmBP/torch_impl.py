import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def baddbmm(input, batch1, batch2):
    ret = torch.baddbmm(input, batch1, batch2)
    ret.backward(ret)
    return ret

def args_adaptor(np_args):
    input = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    batch1 = torch.from_numpy(np_args[1]).to(torch.float32).cuda()
    batch2 = torch.from_numpy(np_args[2]).to(torch.float32).cuda()
    input.requires_grad = True
    batch1.requires_grad = True
    batch2.requires_grad = True
    return [input, batch1, batch2]


def executer_creator():
    return Executer(baddbmm, args_adaptor)
