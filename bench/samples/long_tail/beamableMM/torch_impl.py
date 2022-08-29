import torch
import torch.nn as nn
from long_tail_bench.core.executer import Executer


class BeamableMM(nn.Module):
    """This module provides an optimized MM for beam decoding with attention.

    It leverage the fact that the source-side of the input is replicated beam
    times and the target-side of the input is of width one. This layer speeds
    up inference by replacing the inputs {(bsz x 1 x nhu), (bsz x sz2 x nhu)}
    with smaller inputs {(bsz/beam x beam x nhu), (bsz/beam x sz2 x nhu)}.
    """
    def __init__(self, beam_size=None):
        super(BeamableMM, self).__init__()
        self.beam_size = beam_size

    def forward(self, input1, input2):
        if (not self.training and self.beam_size is not None  # test mode
                and input1.dim() == 3  # beam size is set
                and input1.size(1)  # only support batched input
                == 1):  # single time step update
            bsz, beam = input1.size(0), self.beam_size

            # bsz x 1 x nhu --> bsz/beam x beam x nhu
            input1 = input1[:, 0, :].unfold(0, beam, beam).transpose(2, 1)

            # bsz x sz2 x nhu --> bsz/beam x sz2 x nhu
            input2 = input2.unfold(0, beam, beam)[:, :, :, 0]

            # use non batched operation if bsz = beam
            if input1.size(0) == 1:
                output = torch.mm(input1[0, :, :], input2[0, :, :])
            else:
                output = input1.bmm(input2)
            return output.view(bsz, 1, -1)
        else:
            return input1.bmm(input2)

    def set_beam_size(self, beam_size):
        self.beam_size = beam_size


def args_adaptor(np_args):
    input1 = torch.from_numpy(np_args[0]).cuda()
    input2 = torch.from_numpy(np_args[1]).cuda()
    return [input1, input2]


def executer_creator():
    coder_instance = BeamableMM()
    return Executer(coder_instance.forward, args_adaptor)
