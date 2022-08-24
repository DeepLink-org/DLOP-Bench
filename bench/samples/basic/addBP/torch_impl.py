import torch
import torch.nn
from bench.core.executer import Executer

def addBP(add1, add2):
    ret = torch.add(add1, add2)
    ret.backward(torch.ones_like(ret))
    return ret

def args_adaptor(np_args):
    add1 = torch.tensor(np_args[0], requires_grad=True).to(torch.float32).cuda()
    add2 = torch.tensor(np_args[1], requires_grad=True).to(torch.float32).cuda()
    return [add1, add2]


def executer_creator():
    return Executer(addBP, args_adaptor)
