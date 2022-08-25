import torch
import torch.nn
from bench.core.executer import Executer

def isnan(isnan_0):
    isnan.requires_grad = True
    ret = torch.isnan(isnan_0)
    ret.backward(torch.ones_like(ret))
    return ret

def args_adaptor(np_args):
    isnan_0 = torch.from_numpy(np_args[0]).cuda()
    return [isnan_0]


def executer_creator():
    return Executer(isnan, args_adaptor)
