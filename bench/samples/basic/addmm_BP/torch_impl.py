from math import ceil
from operator import add
import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer

def addmm_(input_image, mat1_image, mat2_image):
    ret = torch.Tensor.addmm_(input_image, mat1_image, mat2_image).cuda()
    ret.backward(ret)
    return ret

def args_adaptor(np_args):
    input_image = np_args[0]
    input_image.requires_grad = True
    mat1_image = np_args[1]
    mat1_image.requires_grad = True
    mat2_image = np_args[2]
    mat2_image.requires_grad = True
    return [input_image, mat1_image, mat2_image]

def executer_creator():
    return Executer(addmm_, args_adaptor)
