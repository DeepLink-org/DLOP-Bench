# Copyright (c) OpenComputeLab. All Rights Reserved.
# Modified from OpenMMLab.

import torch
from bench.core.executer import Executer


class GaussianSigmoidFocalLoss(object):
    def __init__(
        self,
        gamma,
        beta,
        num_classes,
        init_prior,
        name="gaussian_sigmoid_focal_loss",
        reduction="mean",
        loss_weight=1.0,
        ignore_index=-1,
    ):
        """
        Arguments:
            - name (:obj:`str`): name of the loss function
            - reduction (:obj:`str`): reduction type, choice of mean, none, sum
            - loss_weight (:obj:`float`): loss weight
            - gamma (:obj:`float`): hyparam
            - beta (:obj:`float`): hyparam
            - init_prior (:obj:`float`): init bias initialization
            - num_classes (:obj:`int`): num_classes total, 81 for coco
            - ignore index (:obj:`int`): ignore index in label
        """
        # activation_type = "sigmoid"

        self.gamma = gamma
        self.beta = beta
        assert ignore_index == -1, "only -1 is allowed for ignore index"

    def forward(self, input, target, reduction="mean", normalizer=None):
        """
        Arguments:
            - input (FloatTenosor): [B,C,H,W]
            - target (FloatTenosor): [B,C,H,W]
        """
        assert reduction != "none", "Not Supported none reduction yet"
        loss = 0
        pos_inds = target.eq(1)
        neg_inds = target.lt(1)
        neg_weights = torch.pow(1 - target[neg_inds], self.beta)

        input = torch.clamp(input.sigmoid_(), min=1e-4, max=1 - 1e-4)
        pos_pred = input[pos_inds]
        neg_pred = input[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, self.gamma)
        neg_loss = (torch.log(1 - neg_pred) * torch.pow(neg_pred, self.gamma) *
                    neg_weights)

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss


def args_adaptor(np_args):
    input = torch.from_numpy(np_args[0]).cuda()
    input.requires_grad = True
    gt = torch.from_numpy(np_args[1]).cuda()
    return [input, gt]


def executer_creator():
    coder_instance = GaussianSigmoidFocalLoss(2, 4, 81, 0.01)
    return Executer(coder_instance.forward, args_adaptor)
