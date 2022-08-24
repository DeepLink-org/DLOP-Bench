import torch
import torch.nn
import numpy as np
from bench.core.executer import Executer


def margin_ranking_loss(theta, size, align_corners):
    ret = torch.nn.functional.affine_grid(theta, torch.Size(size), align_corners)
    ret.backward(torch.ones_like(ret))
    return ret


def args_adaptor(np_args):
    theta_torch = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    theta_torch.requires_grad = True

    return [theta_torch, np_args[1], np_args[2]]


def executer_creator():
    return Executer(margin_ranking_loss, args_adaptor)
