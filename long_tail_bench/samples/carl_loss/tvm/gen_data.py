import torch
import numpy as np

def gen_np_args(M, N):
    num_class = 80
    cls_score = np.random.rand(M, num_class)
    labels = np.random.randint(num_class, size=M)
    bbox_pred = np.random.rand(M, N)
    bbox_targets = np.random.rand(M, N)

    return [cls_score, labels, bbox_pred, bbox_targets]

def args_adaptor(np_args):
    cls_score = torch.from_numpy(np_args[0]).cuda()
    labels = torch.from_numpy(np_args[1]).cuda()
    bbox_pred = torch.from_numpy(np_args[2]).cuda()
    bbox_targets = torch.from_numpy(np_args[3]).cuda()

    return [cls_score, labels, bbox_pred, bbox_targets]
