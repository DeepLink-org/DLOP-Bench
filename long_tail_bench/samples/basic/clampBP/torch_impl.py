import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def clamp(input_image_np, min, max):
    input_image = torch.from_numpy(input_image_np).to(torch.float32).cuda()
    input_image.requires_grad = True
    ret = torch.clamp(input_image, min, max)
    ret.backward(ret)
    return ret

def args_adaptor(np_args):
    input_image_np = np_args[0]
    min = np_args[1]
    max = np_args[2]
    
    return [input_image_np, min, max]

def executer_creator():
    return Executer(clamp, args_adaptor)
