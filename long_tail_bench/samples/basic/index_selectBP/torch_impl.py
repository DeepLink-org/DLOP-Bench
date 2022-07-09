import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer

def index_select(input_size, dim, index):
    input_image_np = np.random.random(input_size)
    input_image = torch.from_numpy(input_image_np).to(torch.float32).cuda()
    input_image.requires_grad = True
    index_tensor = torch.randint(0, input_size[dim], (int(index[0]),)).cuda()

    ret = torch.index_select(input_image, dim, index_tensor)
    ret.backward(ret)
    return ret

def args_adaptor(np_args):
    input_size = np_args[0]
    dim = int(np_args[1])
    index = np_args[2]
    return [input_size, dim, index]

def executer_creator():
    return Executer(index_select, args_adaptor)
