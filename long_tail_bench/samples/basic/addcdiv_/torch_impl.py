import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def addcdiv_(input_image_np, input_image_np1, input_image_np2):
    input_image = torch.from_numpy(input_image_np).to(torch.float32).cuda()
    input_image1 = torch.from_numpy(input_image_np1).to(torch.float32).cuda()
    input_image2 = torch.from_numpy(input_image_np2).to(torch.float32).cuda()
    ret = input_image.addcdiv_(input_image1, input_image2)
    return ret


def args_adaptor(np_args):
    input_image_np = np_args[0]
    input_image_np1 = np_args[1]
    input_image_np2 = np_args[2]
    
    return [input_image_np, input_image_np1, input_image_np2]


def executer_creator():
    return Executer(addcdiv_, args_adaptor)
