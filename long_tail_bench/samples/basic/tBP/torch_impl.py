import torch
import torch.nn
from long_tail_bench.core.executer import Executer

def tBP(input_torch):
    ret = torch.t(input_torch)
    ret.backward(torch.ones_like(ret))
    return ret

def args_adaptor(np_args):
    input_torch = torch.tensor(np_args[0], requires_grad=True).to(torch.float32).cuda()
    return [input_torch]


def executer_creator():
    return Executer(tBP, args_adaptor)
