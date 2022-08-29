# Copyright (c) OpenComputeLab. All Rights Reserved.

from math import ceil
from operator import add
import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer

def addmm_(input_image, mat1_image, mat2_image):
    ret = torch.Tensor.addmm_(input_image, mat1_image, mat2_image).cuda()
    return ret

def args_adaptor(np_args):
    input_image = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    mat1_image = torch.from_numpy(np_args[1]).to(torch.float32).cuda()
    mat2_image = torch.from_numpy(np_args[2]).to(torch.float32).cuda()

    return [input_image, mat1_image, mat2_image]

def executer_creator():
    return Executer(addmm_, args_adaptor)
