import torch
import torch.nn as nn
from long_tail_bench.core.executer import Executer


def _make_input(t, requires_grad=False, device=torch.device("cpu")):
    """Make zero inputs for AE loss.

    Args:
        t (torch.Tensor): input
        requires_grad (bool): Option to use requires_grad.
        device: torch device
    Returns:
        inp (torch.Tensor): zero input.
    """
    inp = torch.autograd.Variable(t, requires_grad=requires_grad)
    inp = inp.sum()
    inp = inp.to(device)
    return inp


class AELoss(nn.Module):
    """Associative Embedding loss.

    `Associative Embedding: End-to-End Learning for Joint Detection and
    Grouping <https://arxiv.org/abs/1611.05424v2>`
    """
    def __init__(self, loss_type):
        super().__init__()
        self.loss_type = loss_type

    def singleTagLoss(self, pred_tag, joints):
        """Associative embedding loss for one image.

        Note:
            heatmaps weight: W
            heatmaps height: H
            max_num_people: M
            num_keypoints: K
        Args:
            pred_tag(torch.Tensor[(KxHxW)x1]): tag of output for one image.
            joints(torch.Tensor[MxKx2]): joints information for one image.
        """
        tags = []
        pull = 0
        for joints_per_person in joints:
            tmp = []
            for joint in joints_per_person:
                if joint[1] > 0:
                    print("joint[0]: ", joint[0])
                    print("pred_tag: ", pred_tag)
                    tmp.append(pred_tag[joint[0]])
            if len(tmp) == 0:
                continue
            tmp = torch.stack(tmp)
            tags.append(torch.mean(tmp, dim=0))
            pull = pull + torch.mean((tmp - tags[-1].expand_as(tmp))**2)

        num_tags = len(tags)
        if num_tags == 0:
            return (
                _make_input(torch.zeros(1).float(), device=pred_tag.device),
                _make_input(torch.zeros(1).float(), device=pred_tag.device),
            )
        elif num_tags == 1:
            return (
                _make_input(torch.zeros(1).float(), device=pred_tag.device),
                pull / (num_tags),
            )

        tags = torch.stack(tags)

        size = (num_tags, num_tags)
        A = tags.expand(*size)
        B = A.permute(1, 0)

        diff = A - B

        if self.loss_type == "exp":
            diff = torch.pow(diff, 2)
            push = torch.exp(-diff)
            push = torch.sum(push) - num_tags
        elif self.loss_type == "max":
            diff = 1 - torch.abs(diff)
            push = torch.clamp(diff, min=0).sum() - num_tags
        else:
            raise ValueError("Unknown ae loss type")

        push_loss = push / ((num_tags - 1) * num_tags) * 0.5
        pull_loss = pull / (num_tags)

        return push_loss, pull_loss

    def forward(self, tags, joints):
        """Accumulate the tag loss for each image in the batch.

        Note:
            batch_size: N
            heatmaps weight: W
            heatmaps height: H
            max_num_people: M
            num_keypoints: K

        Args:
            tags(torch.Tensor[Nx(KxHxW)x1]): tag channels of output.
            joints(torch.Tensor[NxMxKx2]): joints information.
        """
        pushes, pulls = [], []
        joints = joints.cpu().data.numpy()
        batch_size = tags.size(0)
        for i in range(batch_size):
            push, pull = self.singleTagLoss(tags[i], joints[i])
            pushes.append(push)
            pulls.append(pull)
        return torch.stack(pushes), torch.stack(pulls)


def args_adaptor(np_args):
    pred_tag = torch.from_numpy(np_args[0]).cuda()
    joints = torch.from_numpy(np_args[1]).cuda()
    return [pred_tag, joints]


def executer_creator():
    coder_instance = AELoss("exp")
    return Executer(coder_instance.singleTagLoss, args_adaptor)
