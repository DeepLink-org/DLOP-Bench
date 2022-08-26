# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def adaptive_max_pool3d(input_image_torch, output_size_image, return_indices_image):
    ret = torch.nn.functional.adaptive_max_pool3d(input_image_torch,\
        output_size_image, return_indices_image)
    ret.backward(torch.ones_like(ret))
    return ret

def args_adaptor(np_args):
    input_image_np = np_args[0]
    input_image_torch = torch.from_numpy(input_image_np).to(torch.float32).cuda()
    input_image_torch.requires_grad = True
    output_size = np_args[1]
    return_indices = np_args[2]
    if (len(output_size)) >1:
        output_size_image = tuple(output_size)
    else:
        output_size_image = output_size[0]
    return_indices_image = return_indices[0]
    return [input_image_torch, output_size_image, return_indices_image]

def executer_creator():
    return Executer(adaptive_max_pool3d, args_adaptor)
