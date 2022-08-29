import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer

def index_select(input_image, dim, index_tensor):
    input_image.requires_grad = True
    ret = torch.index_select(input_image, dim, index_tensor)
    ret.backward(ret)
    return ret

def args_adaptor(np_args):
    dim = int(np_args[1])
    input_image = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    index_tensor = torch.from_numpy(np_args[2]).cuda()
    return [input_image, dim, index_tensor]

def executer_creator():
    return Executer(index_select, args_adaptor)
