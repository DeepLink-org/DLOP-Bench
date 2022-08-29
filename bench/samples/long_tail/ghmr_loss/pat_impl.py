import torch
import torch.nn as nn
from long_tail_bench.core.executer import Executer


class GHMRLoss(nn.Module):
    def __init__(self, mu=0.02, bins=10, momentum=0, loss_weight=1.0):
        self.mu = mu
        self.bins = bins
        self.edges = [float(x) / bins for x in range(bins + 1)]
        self.edges[-1] = 1e3
        self.momentum = momentum
        self.loss_weight = loss_weight
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]

    def forward_single(self, input, target, mask):
        mu = self.mu
        edges = self.edges
        mmt = self.momentum

        # ASL1 loss
        diff = input - target
        loss = torch.sqrt(diff * diff + mu * mu) - mu

        # gradient length
        g = torch.abs(diff / torch.sqrt(mu * mu + diff * diff)).detach()
        weights = torch.zeros_like(g)

        valid = mask > 0
        tot = max(mask.float().sum().item(), 1.0)
        n = 0  # n: valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                n += 1
                if mmt > 0:
                    self.acc_sum[i] = (mmt * self.acc_sum[i] +
                                       (1 - mmt) * num_in_bin)
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
        if n > 0:
            weights /= n

        loss = loss * weights
        loss = loss.sum() / tot
        return loss * self.loss_weight


def args_adaptor(np_args):
    bbox = torch.from_numpy(np_args[0]).cuda()
    bbox.requires_grad = True
    target = torch.from_numpy(np_args[1]).cuda()
    mask = torch.from_numpy(np_args[2]).cuda()

    return [bbox, target, mask]


def executer_creator():
    coder_instance = GHMRLoss()
    return Executer(coder_instance.forward_single, args_adaptor)
