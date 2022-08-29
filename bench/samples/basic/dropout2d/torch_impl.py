import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def dropout2d(input_image, p, inplace):
    dropout2d = torch.nn.Dropout2d(p=p, inplace=inplace)
    ret = dropout2d(input_image)
    return ret


def args_adaptor(np_args):
    input_image = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    p = np_args[1]
    inplace = np_args[2]
    
    return [input_image, p, inplace]

def executer_creator():
    return Executer(dropout2d, args_adaptor)
