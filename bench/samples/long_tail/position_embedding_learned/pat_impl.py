import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional
from long_tail_bench.core.executer import Executer


class NestedTensor(object):

    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(num, num_pos_feats)
        self.col_embed = nn.Embedding(num, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):

        x = tensor_list.tensors

        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = (torch.cat(
            [
                x_emb.unsqueeze(0).repeat(h, 1, 1),
                y_emb.unsqueeze(1).repeat(1, w, 1),
            ],
            dim=-1,
        ).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1))
        return pos


def args_adaptor(np_args):
    x = torch.from_numpy(np_args[0]).cuda()
    mask = torch.from_numpy(np_args[1]).cuda() > 0.5
    tensor_list = NestedTensor(x, mask)
    return [tensor_list]


def executer_creator():
    coder_instance = PositionEmbeddingLearned(50, 128).cuda()
    return Executer(coder_instance.forward,
                    args_adaptor).register_custom_class(NestedTensor)
