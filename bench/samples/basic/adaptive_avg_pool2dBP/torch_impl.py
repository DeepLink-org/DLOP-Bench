import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer

def adaptive_avg_pool2d(input_image, output_size_image):
    ret = torch.nn.functional.adaptive_avg_pool2d(input_image, output_size_image)
    ret.backward(ret)
    return ret

def args_adaptor(np_args):
    input_image_np = np_args[0]
    output_size = np_args[1]
    input_image = torch.from_numpy(input_image_np).to(torch.float32).cuda()
    input_image.requires_grad = True

    if (len(output_size)) >1:
        output_size_image = tuple(output_size)
    else:
        output_size_image = output_size[0]
    return [input_image, output_size_image]

def executer_creator():
    return Executer(adaptive_avg_pool2d, args_adaptor)