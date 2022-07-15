import torch
import torch.nn
from long_tail_bench.core.executer import Executer

def sigmoid(sigmoid_0):
    sigmoid.requires_grad = True
    ret = torch.sigmoid(sigmoid_0)
    ret.backward(ret)
    return ret

def args_adaptor(np_args):
    sigmoid_0 = torch.from_numpy(np_args[0]).cuda()
    return [sigmoid_0]


def executer_creator():
    return Executer(sigmoid, args_adaptor)
