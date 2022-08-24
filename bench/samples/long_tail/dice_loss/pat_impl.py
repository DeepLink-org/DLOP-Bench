import torch
from bench.core.executer import Executer


class DiceLoss(object):
    def __init__(
        self,
        act="softmax",
        squared_pred=True,
        jaccard=True,
    ):
        self.act = act
        self.squared_pred = squared_pred
        self.jaccard = jaccard

    def forward(self, pred, gt, smooth=1e-5):
        # gt = gt.long()
        if self.act == "sigmoid":
            pred = torch.sigmoid(pred)
        elif self.act == "softmax":
            pred = torch.softmax(pred, 1)
        else:
            raise ("unsuport activation")

        if len(pred.shape) != len(gt.shape):
            gt.unsqueeze_(1)
        if pred.shape[1] != gt.shape[1]:
            one_hot = torch.zeros_like(pred).to(pred.device)
            one_hot.scatter_(1, gt, 1)
            gt = one_hot

        reduce_axis = list(range(2, len(pred.shape)))

        intersection = torch.sum(gt * pred, reduce_axis)

        if self.squared_pred:
            gt = torch.pow(gt, 2)
            pred = torch.pow(pred, 2)

        gt_sum = torch.sum(gt, reduce_axis)
        pred_sum = torch.sum(pred, reduce_axis)

        denominator = gt_sum + pred_sum

        if self.jaccard:
            denominator -= intersection

        dice = (2.0 * intersection + smooth) / (denominator + smooth)
        return 1.0 - dice.mean()


def args_adaptor(np_args):
    boxes = torch.from_numpy(np_args[0]).cuda()
    boxes.requires_grad = True
    gt = torch.from_numpy(np_args[1]).cuda()
    gt.requires_grad = True
    return [boxes, gt]


def executer_creator():
    coder_instance = DiceLoss()
    return Executer(coder_instance.forward, args_adaptor)
