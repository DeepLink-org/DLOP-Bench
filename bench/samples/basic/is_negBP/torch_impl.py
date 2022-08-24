import torch
import torch.nn
from bench.core.executer import Executer

def is_neg(is_neg_0):
    is_neg.requires_grad = True
    ret = torch.is_neg(is_neg_0)
    ret.backward(ret)
    return ret

def args_adaptor(np_args):
    is_neg_0 = torch.from_numpy(np_args[0]).cuda()
    return [is_neg_0]


def executer_creator():
    return Executer(is_neg, args_adaptor)
