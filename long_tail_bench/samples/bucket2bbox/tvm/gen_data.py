import torch
import numpy as np

def gen_np_args(M):
    proposals = np.random.rand(M, 4)
    cls_preds = np.random.rand(M, 16)
    offset_preds = np.random.rand(M, 16)
    return [proposals, cls_preds, offset_preds]

def args_adaptor(np_args):
    proposals = torch.from_numpy(np_args[0]).cuda()
    cls_preds = torch.from_numpy(np_args[1]).cuda()
    offset_preds = torch.from_numpy(np_args[2]).cuda()

    return [proposals, cls_preds, offset_preds]
