import torch
import torch.nn as nn

from long_tail_bench.core.executer import Executer


class JointsMSELoss(nn.Module):
    """MSE loss for heatmaps.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
    """
    def __init__(self, use_target_weight=True):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        """Forward function."""
        batch_size = output.size(0)
        num_joints = output.size(1)

        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0.0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)
            if self.use_target_weight:
                loss += self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx]),
                )
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


def args_adaptor(np_args):
    bboxes = torch.from_numpy(np_args[0]).cuda()
    bboxes.requires_grad = True
    gt = torch.from_numpy(np_args[1]).cuda()
    target_weight = torch.from_numpy(np_args[2]).cuda()

    return [bboxes, gt, target_weight]


def executer_creator():
    coder_instance = JointsMSELoss()
    return Executer(coder_instance.forward, args_adaptor)
