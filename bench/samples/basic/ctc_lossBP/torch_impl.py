# Copyright (c) OpenComputeLab. All Rights Reserved.

import torch
from torch.nn import functional
from bench.core.executer import Executer


def ctc_loss(log_probs, targets, input_lengths,
            target_lengths, blank, reduction, zero_infinity):
    output = functional.ctc_loss(log_probs, targets, input_lengths,
            target_lengths, blank, reduction, zero_infinity)
    output.backward(output)    
    return output


def args_adaptor(np_args):
    log_probs = torch.from_numpy(np_args[0]).cuda()
    targets = torch.from_numpy(np_args[1]).cuda()
    input_lengths = torch.from_numpy(np_args[2]).cuda()
    target_lengths = torch.from_numpy(np_args[3]).cuda()
    log_probs.requires_grad = True
    return [log_probs, targets, input_lengths,
            target_lengths, np_args[4], np_args[5], np_args[6]]


def executer_creator():
    return Executer(ctc_loss, args_adaptor)
