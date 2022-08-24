import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer



def where(condition, tensor1, tensor2):
    ret = torch.where(condition, tensor1, tensor2)
    ret.backward(torch.ones_like(ret))
    return ret


def args_adaptor(np_args):
    condition_torch = torch.tensor(np_args[0]).to(torch.bool).cuda()
    tensor1_torch = torch.tensor(np_args[1], requires_grad=True).to(torch.float32).cuda()
    tensor2_torch = torch.tensor(np_args[2], requires_grad=True).to(torch.float32).cuda()
    return [condition_torch, tensor1_torch, tensor2_torch]


def executer_creator():
    return Executer(where, args_adaptor)
