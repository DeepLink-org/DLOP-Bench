import torch
import torch.nn as nn
import torch.nn.functional as F
from long_tail_bench.core.executer import Executer


class SmoothLoss(nn.Module):
    label_keys = ["edge_logit", "edge_target"]

    def __init__(
        self,
        loss_weight=10.0,
        reduction="mean",
        beta=10.0,
        k=3,
        m=3,
        weight_type="gs",
    ):
        super(SmoothLoss, self).__init__()
        self.loss_weight = loss_weight
        assert reduction in ["mean", "sum"]
        self.reduction = reduction
        self.beta = beta
        self.k = k
        self.m = m
        self.weight_type = weight_type

    def forward(self,
                logit,
                edge_logit,
                edge_target,
                ignore_index=255,
                **kwargs):
        beta = self.beta
        k, m = self.k, self.m
        temp_type = self.weight_type

        edge_target = torch.unsqueeze(edge_target.clone(), dim=1).float()
        edge_target[edge_target != 1] = 0
        kernel = torch.Tensor([[1.0] * k for i in range(k)]).view((1, 1, k, k))
        edge_target = F.conv2d(edge_target,
                               kernel.type_as(edge_target),
                               padding=k >> 1)
        edge_target[edge_target > m] = m
        # edge_target /= self.m
        edge_pred = torch.sigmoid(edge_logit).detach().clone()

        if "gs" in temp_type:
            edge_weight = edge_target
        elif "ps" in temp_type:
            edge_weight = edge_pred
        else:
            assert "gps" in temp_type
            edge_weight = (edge_pred + edge_target) / 2

        pred = F.pad(F.softmax(logit, dim=1),
                     pad=[0, 1, 0, 1],
                     mode="replicate")
        grad_x = torch.abs(pred[:, :, :-1, 1:] - pred[:, :, :-1, :-1])
        grad_y = torch.abs(pred[:, :, 1:, :-1] - pred[:, :, :-1, :-1])

        weights = torch.exp(-beta * torch.abs(edge_weight))
        grad_x *= weights.type_as(grad_x)
        grad_y *= weights.type_as(grad_y)
        loss = grad_x.mean() + grad_y.mean()

        return self.loss_weight * loss

    def extra_repr(self):
        s = f"lossweight={self.loss_weight}, beta={self.beta}, k={self.k}, \
              m={self.m}, weight_type={self.weight_type}"

        return s


def args_adaptor(np_args):
    logit = torch.from_numpy(np_args[0]).cuda()
    logit.requires_grad = True
    edge_logit = torch.from_numpy(np_args[1]).cuda()
    edge_logit.requires_grad = True
    label = torch.from_numpy(np_args[2]).cuda()
    label = torch.argmax(label, dim=1)
    return [logit, edge_logit, label]


def executer_creator():
    coder_instance = SmoothLoss().cuda()
    return Executer(coder_instance.forward, args_adaptor)
