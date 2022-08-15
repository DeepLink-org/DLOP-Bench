from math import ceil
import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer

def true_divide(input_image, other_image):
    ret = torch.true_divide(input_image, other_image)
    ret.backward(ret)
    return ret

def args_adaptor(np_args):
    input_image = np_args[0]
    input_image.requires_grad = True
    other = np_args[1]
    return [input_image, other]

def executer_creator():
    return Executer(true_divide, args_adaptor)