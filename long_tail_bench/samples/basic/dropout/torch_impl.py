import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def dropout(input_image_np, p, inplace):
    input_image = torch.from_numpy(input_image_np).to(torch.float32).cuda()
    dropout = torch.nn.Dropout(p=p, inplace=inplace)
    ret = dropout(input_image)
    return ret


def args_adaptor(np_args):
    input_image_np = np_args[0]
    p = np_args[1]
    inplace = np_args[2]
    
    return [input_image_np, p, inplace]

def executer_creator():
    return Executer(dropout, args_adaptor)
