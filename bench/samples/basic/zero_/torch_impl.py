# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import numpy as np
from bench.core.executer import Executer

def zero_(input_image):
    ret = torch.zero_(input_image)
    return ret

def args_adaptor(np_args):
    size = np_args[0]
    input_image_np = np.random.random(size)
    input_image = torch.from_numpy(input_image_np).to(torch.float32).cuda()
    return [input_image]

def executer_creator():
    return Executer(zero_, args_adaptor)