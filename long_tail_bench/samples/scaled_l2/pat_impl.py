import torch
from long_tail_bench.core.executer import Executer


def scaled_l2(x, codewords, scale):
    num_codes, channels = codewords.size()
    batch_size = x.size(0)
    reshaped_scale = scale.view((1, 1, num_codes))
    expanded_x = x.unsqueeze(2).expand(
        (batch_size, x.size(1), num_codes, channels))
    reshaped_codewords = codewords.view((1, 1, num_codes, channels))

    scaled_l2_norm = reshaped_scale * (expanded_x -
                                       reshaped_codewords).pow(2).sum(dim=3)
    return scaled_l2_norm


def args_adaptor(np_args):
    x = torch.from_numpy(np_args[0]).cuda()
    x.requires_grad = True
    codewords = torch.from_numpy(np_args[1]).cuda()
    codewords.requires_grad = True
    scale = torch.from_numpy(np_args[2]).cuda()
    scale.requires_grad = True

    return [x, codewords, scale]


def executer_creator():
    return Executer(scaled_l2, args_adaptor)
