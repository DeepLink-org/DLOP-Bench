# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def addcmul_(input_image, input_image1, input_image2):
    ret = input_image.addcmul_(input_image1, input_image2)
    return ret


def args_adaptor(np_args):
    input_image = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    input_image1 = torch.from_numpy(np_args[1]).to(torch.float32).cuda()
    input_image2 = torch.from_numpy(np_args[2]).to(torch.float32).cuda()
    
    return [input_image, input_image1, input_image2]


def executer_creator():
    return Executer(addcmul_, args_adaptor)
