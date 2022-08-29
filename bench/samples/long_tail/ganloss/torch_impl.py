# Copyright (c) OpenComputeLab. All Rights Reserved.
# Modified from OpenMMLab.
import torch
import torch.nn as nn
from bench.core.executer import Executer


class GANLoss(nn.Module):
    def __init__(self,
                 gan_type,
                 real_label_val=1.0,
                 fake_label_val=0.0,
                 loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == "lsgan":
            self.loss = nn.MSELoss()
        elif self.gan_type == "wgan":
            self.loss = self._wgan_loss
        elif self.gan_type == "hinge":
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(
                f"GAN type {self.gan_type} is not implemented.")

    def _wgan_loss(self, input, target):

        return -input.mean() if target else input.mean()

    def get_target_label(self, input, target_is_real):

        if self.gan_type == "wgan":
            return target_is_real
        target_val = (self.real_label_val
                      if target_is_real else self.fake_label_val)

        # For parrots jit: new_ones empty op shape is fixed
        # because of python value input.size()
        return input.ones_like(input) * target_val
        # return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real=True, is_disc=False):

        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == "hinge":
            if is_disc:
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:
                loss = -input.mean()
        else:
            loss = self.loss(input, target_label)

        return loss if is_disc else loss * self.loss_weight


def args_adaptor(np_args):
    inputs = torch.from_numpy(np_args[0]).cuda()
    return [inputs]


def vanilla_executer_creator():
    coder_instance = GANLoss("vanilla",
                             loss_weight=2.0,
                             real_label_val=1.0,
                             fake_label_val=0.0)
    return Executer(coder_instance.forward, args_adaptor)


def lsgan_executer_creator():
    coder_instance = GANLoss("lsgan",
                             loss_weight=2.0,
                             real_label_val=1.0,
                             fake_label_val=0.0)
    return Executer(coder_instance.forward, args_adaptor)


def wgan_executer_creator():
    coder_instance = GANLoss("wgan",
                             loss_weight=2.0,
                             real_label_val=1.0,
                             fake_label_val=0.0)
    return Executer(coder_instance.forward, args_adaptor)


def hinge_executer_creator():
    coder_instance = GANLoss("hinge",
                             loss_weight=2.0,
                             real_label_val=1.0,
                             fake_label_val=0.0)
    return Executer(coder_instance.forward, args_adaptor)
