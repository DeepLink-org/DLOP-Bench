import torch
import torch.nn as nn
import torch.nn.functional as F
from long_tail_bench.core.executer import Executer


class GradientLoss(nn.Module):
    label_keys = ["gt_soft_seg", "gt_semantic_seg"]

    def __init__(self, loss_weight=1.0, ignore_index=255, num_classes=2):
        super(GradientLoss, self).__init__()
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        sobel_kernel_x = (torch.Tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0],
                                        [1.0, 0.0, -1.0]]).view(
                                            (1, 1, 3, 3)).expand(
                                                (num_classes, 1, 3, 3)))
        sobel_kernel_y = (torch.Tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0],
                                        [-1.0, -2.0, -1.0]]).view(
                                            (1, 1, 3, 3)).expand(
                                                (num_classes, 1, 3, 3)))
        self.sobel_x = nn.Conv2d(
            num_classes,
            num_classes,
            3,
            stride=1,
            padding=1,
            groups=num_classes,
            bias=False,
        )
        self.sobel_y = nn.Conv2d(
            num_classes,
            num_classes,
            3,
            stride=1,
            padding=1,
            groups=num_classes,
            bias=False,
        )
        self.sobel_x.weight = nn.Parameter(sobel_kernel_x, requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_kernel_y, requires_grad=False)
        self.sobel_x = self.sobel_x.cuda()
        self.sobel_y = self.sobel_y.cuda()

    def forward(self, preds_S, preds_T, label, **kwargs):
        """Forward function."""
        if (preds_S.shape[-1] != preds_T.shape[-1]
                or preds_S.shape[-2] != preds_T.shape[-2]):
            preds_S = F.interpolate(preds_S,
                                    size=preds_T.shape[-2:],
                                    mode="bilinear")
        if len(label.shape) == 3:
            label = label.unsqueeze(1)
        if (label.shape[-1] != preds_T.shape[-1]
                or label.shape[-2] != preds_T.shape[-2]):
            label = F.interpolate(label.float(),
                                  size=preds_T.shape[-2:],
                                  mode="nearest").expand(preds_T.size())
        index = label != self.ignore_index
        if not torch.any(index):
            return preds_S.mean() * 0
        index = index.expand(preds_T.size())
        preds_S_grad_x = self.sobel_x(preds_S)
        preds_S_grad_y = self.sobel_y(preds_S)
        preds_S_magnitude = torch.sqrt(
            torch.pow(preds_S_grad_x, 2) + torch.pow(preds_S_grad_y, 2) + 1e-5)
        preds_S_grad_x = preds_S_grad_x / (preds_S_magnitude)
        preds_S_grad_y = preds_S_grad_y / (preds_S_magnitude)

        preds_T_grad_x = self.sobel_x(preds_T)
        preds_T_grad_y = self.sobel_y(preds_T)
        preds_T_magnitude = torch.sqrt(
            torch.pow(preds_T_grad_x, 2) + torch.pow(preds_T_grad_y, 2) + 1e-5)
        preds_T_grad_x = preds_T_grad_x / (preds_T_magnitude)
        preds_T_grad_y = preds_T_grad_y / (preds_T_magnitude)

        mse = nn.MSELoss(reduction="none")
        loss1 = mse(preds_S_grad_x, preds_T_grad_x)
        loss2 = mse(preds_S_grad_y, preds_T_grad_y)
        loss = loss1[index].mean() + loss2[index].mean()

        return self.loss_weight * loss

    def extra_repr(self):
        s = f"lossweight={self.loss_weight}"
        return s


def args_adaptor(np_args):
    logit = torch.from_numpy(np_args[0]).cuda()
    logit.requires_grad = True

    label = torch.from_numpy(np_args[1]).cuda()
    label = torch.argmax(label, dim=1)

    soft_label = torch.from_numpy(np_args[2]).cuda()

    return [logit, soft_label, label]


def executer_creator():
    coder_instance = GradientLoss().cuda()
    return Executer(coder_instance.forward, args_adaptor)
