# Copyright (c) OpenComputeLab. All Rights Reserved.

from math import ceil
import torch
import torch.nn
import numpy as np
import operator
from bench.core.executer import Executer

def itruediv(input_image, other_image):
    ret = operator.itruediv(input_image, other_image)
    ret.backward(torch.ones_like(ret))
    return ret

def args_adaptor(np_args):
    input_image = np_args[0]
    other = np_args[1]
    input_image.requires_grad = True

    if (len(other)) >1 :
        other_image = tuple(other)
    else:
        other_image = other[0]
        
    return [input_image, other_image]

def executer_creator():
    return Executer(itruediv, args_adaptor)