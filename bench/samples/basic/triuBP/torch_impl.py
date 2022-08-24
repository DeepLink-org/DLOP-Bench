from math import ceil
from operator import add
import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer

def triu(input_image, out_image):
    ret = torch.triu(input_image, diagonal=out_image).cuda()
    ret.backward(ret)
    return ret

def args_adaptor(np_args):
    input_image = np_args[0]
    input_image.requires_grad = True
    out_ = np_args[1]
    out_image = out_[0]

    return [input_image, out_image]

def executer_creator():
    return Executer(triu, args_adaptor)
