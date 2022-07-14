import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def clamp(input_image, min, max):
    ret = torch.clamp(input_image, min, max)
    return ret

def args_adaptor(np_args):
    input_image = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    min = np_args[1]
    max = np_args[2]
    
    return [input_image, min, max]

def executer_creator():
    return Executer(clamp, args_adaptor)
