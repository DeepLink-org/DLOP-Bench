import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer

def adaptive_avg_pool2d(input_image, output_size_image):
    ret = torch.nn.functional.adaptive_avg_pool2d(input_image, output_size_image)
    return ret

def args_adaptor(np_args):
    input_image = np_args[0]
    output_size = np_args[1]
    if (len(output_size)) >1:
        output_size_image = tuple(output_size)
    else:
        output_size_image = output_size[0]
    return [input_image, output_size_image]

def executer_creator():
    return Executer(adaptive_avg_pool2d, args_adaptor)
