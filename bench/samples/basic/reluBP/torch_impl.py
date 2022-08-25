import torch
import torch.nn
from bench.core.executer import Executer

def relu(relu_0):
    relu.requires_grad = True
    ret = torch.relu(relu_0)
    ret.backward(torch.ones_like(ret))
    return ret

def args_adaptor(np_args):
    relu_0 = torch.from_numpy(np_args[0]).cuda()
    return [relu_0]


def executer_creator():
    return Executer(relu, args_adaptor)
