from math import ceil
from operator import add

from requests import TooManyRedirects
import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer

def hstack(input_image, out_image):
    # TODO
    # ERROR arguments from given CSV 
    # hstack,torch,"[[torch.Size([32])]], [[torch.Size([32])]], [[torch.Size([32])]], [[torch.Size([32])]], [[torch.Size([32])]]","[{}, {}, {}, {}, {}]",5
    
    # CORRECT input: sequence of tensors to concatenate
    ret = torch.hstack((input_image, out_image))
    return ret

def args_adaptor(np_args):
    input_image = np_args[0]
    out_image = np_args[1]

    return [input_image, out_image]

def executer_creator():
    return Executer(hstack, args_adaptor)