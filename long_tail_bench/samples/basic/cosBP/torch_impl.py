import torch
import torch.nn
from long_tail_bench.core.executer import Executer

def cos(cos_0):
    cos.requires_grad = True
    ret = torch.cos(cos_0)
    ret.backward(ret)
    return ret

def args_adaptor(np_args):
    cos_0 = torch.from_numpy(np_args[0]).cuda()
    return [cos_0]


def executer_creator():
    return Executer(cos, args_adaptor)
