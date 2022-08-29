# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
import numpy as np
from bench.core.executer import Executer


def multinomial(input_tensor, num_samples):
    ret = torch.multinomial(input_tensor, num_samples)
    return ret

def args_adaptor(np_args):
    np_input = np_args[0]
    input_tensor = torch.from_numpy(np_input).to("cuda")
    num_samples = np_args[1]
    return [input_tensor, num_samples]

def executer_creator():
    return Executer(multinomial, args_adaptor)