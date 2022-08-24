import torch
import torch.nn as nn

from bench.core.executer import Executer


class HeatmapLoss(nn.Module):
    """Accumulate the heatmap loss for each image in the batch."""
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, mask):
        """
        Note:
            batch_size: N
            heatmaps weight: W
            heatmaps height:
            max_num_people: M
            num_keypoints: K
        Args:
            pred(torch.Tensor[NxKxHxW]):heatmap of output.
            gt(torch.Tensor[NxKxHxW]): target heatmap.
            mask(torch.Tensor[NxHxW]): mask of target.
        """
        assert pred.size() == gt.size()
        loss = ((pred - gt)**2) * mask[:, None, :, :].expand_as(pred)
        loss = loss.mean(dim=3).mean(dim=2).mean(dim=1)
        return loss


def args_adaptor(np_args):
    pred = torch.from_numpy(np_args[0]).cuda()
    pred.requires_grad = True
    gt = torch.from_numpy(np_args[1]).cuda()
    mask = torch.from_numpy(np_args[2]).cuda()

    return [pred, gt, mask]


def executer_creator():
    coder_instance = HeatmapLoss().cuda()
    return Executer(coder_instance.forward, args_adaptor)
