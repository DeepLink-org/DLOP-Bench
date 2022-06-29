import torch
from torch import Tensor
import torch.nn as nn
import math
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


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to
    the one used by the Attention is all you need paper, generalized to work
    on images.
    """

    def __init__(self,
                 num_pos_feats=64,
                 temperature=10000,
                 normalize=False,
                 scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats,
                             dtype=torch.float32,
                             device=x.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


def args_adaptor(np_args):
    x = torch.from_numpy(np_args[0]).cuda()
    mask = torch.from_numpy(np_args[1]).cuda() > 0.5
    tensor_list = NestedTensor(x, mask)
    return [tensor_list]


def executer_creator():
    coder_instance = PositionEmbeddingSine(128, 10000, True)
    return Executer(coder_instance.forward,
                    args_adaptor).register_custom_class(NestedTensor)
