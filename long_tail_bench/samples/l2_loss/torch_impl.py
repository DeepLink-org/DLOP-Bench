import torch
from long_tail_bench.core.executer import Executer


def l2_loss(input, target):
    return torch.mean((input - target) * (input - target))


def args_generator(np_args):
    # output = torch.from_numpy(np_args[0]).cuda()
    # target = torch.from_numpy(np_args[1]).cuda()
    output = torch.randn((16, 32), device='cuda')
    target = torch.randn((16, 32), device='cuda')
    output.requires_grad = True
    return [output, target]


def executer_creator():
    return Executer(l2_loss, args_generator)
