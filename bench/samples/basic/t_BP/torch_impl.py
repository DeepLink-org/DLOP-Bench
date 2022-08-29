import torch
import torch.nn
from long_tail_bench.core.executer import Executer

def t_BP(input_torch):
    output = torch.Tensor.t_(input_torch)
    output.backward(torch.ones_like(output))
    return output

def args_adaptor(np_args):
    input_torch = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    input_torch.requires_grad = True
    return [input_torch]


def executer_creator():
    return Executer(t_BP, args_adaptor)
