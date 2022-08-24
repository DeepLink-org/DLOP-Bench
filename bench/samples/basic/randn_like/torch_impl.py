import torch
import numpy as np
from bench.core.executer import Executer


def randn_like(input_tensor):
    ret = torch.randn_like(input_tensor, device=torch.device("cuda"))
    return ret

def args_adaptor(np_args):
    np_input = np_args[0]
    input_tensor = torch.from_numpy(np_input).to(torch.float32).cuda()
    return [input_tensor]

def executer_creator():
    return Executer(randn_like, args_adaptor)