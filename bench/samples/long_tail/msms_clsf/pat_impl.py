import torch
import torch.nn as nn
import torch.nn.functional as F
from bench.core.executer import Executer


class MSMSclsf(object):
    def get_loss(self, predict, label, loss_type="s-softmax", normalizer=1):
        if loss_type in ["softmax", "a-softmax"]:
            label = label.long()
            criterion = nn.CrossEntropyLoss(ignore_index=-1)
            loss = criterion(predict, label)
        elif loss_type in ["SmoothL1WithThrd"]:
            assert label.shape[1] == 2
            # gt = label[:, 0]
            # thrd = (
            #     label[:, 1]
            #     if label.shape[1] == 2
            #     else torch.Tensor([0]).expand_as(label)
            # )
            # loss = SmoothL1WithThrd(predict, gt, thrd)
        elif loss_type in ["s-softmax"]:
            batch_size = predict.size(0)
            num_class = predict.size(1)
            label_smooth = torch.zeros((batch_size, num_class)).cuda()
            label_smooth.scatter_(dim=1, index=label.unsqueeze(-1), src=1)
            ones_idx = label_smooth == 1
            zeros_idx = label_smooth == 0
            label_smooth[ones_idx] = 0.9
            label_smooth[zeros_idx] = 0.1 / (num_class - 1)
            loss = (-torch.sum(
                F.log_softmax(predict, 1, dtype=torch.int) * label_smooth) /
                    batch_size)
        elif loss_type in ["softmax-comp"]:
            batch_size = int(predict.shape[0] / 2)
            predict_tar = predict[:batch_size]
            predict_base = predict[batch_size:]
            predict_diff = predict_base - predict_tar
            label_half = label[:batch_size].long()
            criterion = nn.CrossEntropyLoss(ignore_index=-1)
            loss = criterion(predict_diff, label_half)
        else:
            raise NotImplementedError

        return loss * normalizer


def args_adaptor(np_args):
    predict = torch.from_numpy(np_args[0]).cuda()
    label = torch.from_numpy(np_args[1]).cuda()
    predict.requires_grad = True
    return [predict, label]


def executer_creator():
    coder_instance = MSMSclsf()
    return Executer(coder_instance.get_loss, args_adaptor)
