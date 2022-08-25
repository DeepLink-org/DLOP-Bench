import torch
import torch.nn
from bench.core.executer import Executer

def is_conj(is_conj_0):
    is_conj.requires_grad = True
    ret = torch.is_conj(is_conj_0)
    ret.backward(torch.ones_like(ret))
    return ret

def args_adaptor(np_args):
    is_conj_0 = torch.from_numpy(np_args[0]).cuda()
    return [is_conj_0]


def executer_creator():
    return Executer(is_conj, args_adaptor)
