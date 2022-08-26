# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer

def adaptive_avg_pool3d(input_image_torch, output_size_image):
    ret = torch.nn.functional.adaptive_avg_pool3d(input_image_torch, output_size_image)
    ret.backward(torch.ones_like(ret))
    return ret

def args_adaptor(np_args):
    output_size = np_args[1]
    input_image_torch = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    input_image_torch.requires_grad = True

    if (len(output_size)) >1:
        output_size_image = tuple(output_size)
    else:
        output_size_image = output_size[0]

    return [input_image_torch, output_size_image]

def executer_creator():
    return Executer(adaptive_avg_pool3d, args_adaptor)
