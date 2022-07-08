import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def clamp(input_size, min, max):
    input_image_np = np.random.random([input_size])
    input_image = torch.from_numpy(input_image_np).to(torch.float32).cuda()
    clamp = torch.clamp(input=input_size, min=min, max=max)
    ret = clamp(input_image)
    return ret

def args_adaptor(np_args):
    input_size = np_args[0]
    min = np_args[1][0]
    max = np_args[2][0]
    
    return [input_size, min, max]

def executer_creator():
    return Executer(clamp, args_adaptor)
