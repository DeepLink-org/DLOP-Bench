import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer
from torch import _C

def adaptive_max_pool3d_with_indices(input_image, output_size_image):
    res, indices = torch.nn.functional.adaptive_max_pool3d_with_indices(\
        input_image, output_size_image)
    res.backward(res)
    return res

def args_adaptor(np_args):
    input_image_np = np_args[0]
    input_image = torch.from_numpy(input_image_np).to(torch.float32).cuda()
    input_image.requires_grad = True
    output_size = np_args[1]
    if (len(output_size)) >1:
        output_size_image = tuple(output_size)
    else:
        output_size_image = output_size[0]
    return [input_image, output_size_image]

def executer_creator():
    return Executer(adaptive_max_pool3d_with_indices, args_adaptor)