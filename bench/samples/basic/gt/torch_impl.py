import torch
import torch.nn
from bench.core.executer import Executer

def gt(gt_0, gt_1):
    return torch.gt(gt_0, gt_1)

def args_adaptor(np_args):
    gt_0 = torch.from_numpy(np_args[0]).cuda()
    gt_1 = torch.from_numpy(np_args[1]).cuda()
    return [gt_0, gt_1]


def executer_creator():
    return Executer(gt, args_adaptor)
