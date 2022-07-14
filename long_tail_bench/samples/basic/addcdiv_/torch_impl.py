import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def addcdiv_(input_image, input_image1, input_image2):
    ret = input_image.addcdiv_(input_image1, input_image2)
    return ret


def args_adaptor(np_args):
    input_image = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    input_image1 = torch.from_numpy(np_args[1]).to(torch.float32).cuda()
    input_image2 = torch.from_numpy(np_args[2]).to(torch.float32).cuda()
    
    return [input_image, input_image1, input_image2]


def executer_creator():
    return Executer(addcdiv_, args_adaptor)
