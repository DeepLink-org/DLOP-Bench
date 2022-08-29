import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer

def pad(input, pad):
    ret = torch.nn.functional.pad(input, pad)
    ret.backward(ret)
    return ret


def args_adaptor(np_args):
    input_image = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    input_image.requires_grad = True
    return [input_image, np_args[1]]

def executer_creator():
    return Executer(pad, args_adaptor)
