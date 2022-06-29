import torch
from long_tail_bench.core.executer import Executer


def aggregate(assigment_weights, x, codewords):
    num_codes, channels = codewords.size()
    reshaped_codewords = codewords.view((1, 1, num_codes, channels))
    batch_size = x.size(0)

    expanded_x = x.unsqueeze(2).expand(
        (batch_size, x.size(1), num_codes, channels))
    encoded_feat = (assigment_weights.unsqueeze(3) *
                    (expanded_x - reshaped_codewords)).sum(dim=1)
    return encoded_feat


def args_adaptor(np_args):
    assigment_weights = torch.from_numpy(np_args[0]).cuda()
    assigment_weights.requires_grad = True
    x = torch.from_numpy(np_args[1]).cuda()
    x.requires_grad = True
    codewords = torch.from_numpy(np_args[2]).cuda()
    codewords.requires_grad = True
    return [assigment_weights, x, codewords]


def executer_creator():
    return Executer(aggregate, args_adaptor)
