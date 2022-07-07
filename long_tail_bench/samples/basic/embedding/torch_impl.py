import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def embedding(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse):
    input_image_np = np.random.random(input)
    input_image = torch.from_numpy(input_image_np).to(torch.long).cuda()
    weight_image_np = np.random.random(weight)
    weight_image = torch.from_numpy(weight_image_np).to(torch.float32).cuda()    
    padding_idx_image = padding_idx[0]
    max_norm_image = max_norm[0]
    norm_type_image = norm_type[0]
    scale_grad_by_freq_image = scale_grad_by_freq[0]
    sparse_iamge = sparse[0]
    ret = torch.nn.functional.embedding(input_image, weight_image, padding_idx_image, max_norm_image, norm_type_image, scale_grad_by_freq_image, sparse_iamge).cuda()
    return ret


def args_adaptor(np_args):
    input = np_args[0]
    weight = np_args[1]
    padding_idx = np_args[2]
    max_norm = np_args[3]
    norm_type = np_args[4] 
    scale_grad_by_freq = np_args[5]
    sparse = np_args[6]
    return [input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse]


def executer_creator():
    return Executer(embedding, args_adaptor)
