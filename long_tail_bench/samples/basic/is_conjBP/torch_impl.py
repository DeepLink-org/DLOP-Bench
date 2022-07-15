import torch
import torch.nn
from long_tail_bench.core.executer import Executer

def is_conj(is_conj_0):
    is_conj.requires_grad = True
    ret = torch.is_conj(is_conj_0)
    ret.backward(ret)
    return ret

def args_adaptor(np_args):
    is_conj_0 = torch.from_numpy(np_args[0])
    return [is_conj_0]


def executer_creator():
    return Executer(is_conj, args_adaptor)
