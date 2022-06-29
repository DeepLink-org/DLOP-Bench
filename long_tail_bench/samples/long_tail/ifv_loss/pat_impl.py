import torch
import torch.nn as nn
import torch.nn.functional as F
from long_tail_bench.core.executer import Executer


class CriterionIFV(nn.Module):
    label_keys = ["gt_soft_seg", "gt_semantic_seg"]

    def __init__(self,
                 classes,
                 loss_weight=1.0,
                 ignore_index=255,
                 mean_mode=0):
        super(CriterionIFV, self).__init__()
        self.num_classes = classes
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.mean_dims = (-1, -2) if mean_mode == 0 else (0, -1, -2)

    def forward(self, preds_S, preds_T, target, *args, **kwargs):
        if (preds_S.shape[-1] != preds_T.shape[-1]
                or preds_S.shape[-2] != preds_T.shape[-2]):
            preds_S = F.interpolate(preds_S,
                                    size=preds_T.shape[-2:],
                                    mode="bilinear")
        feat_S = preds_S
        feat_T = preds_T.detach()
        size_f = (feat_S.shape[2], feat_S.shape[3])
        tar_feat_S = nn.Upsample(size_f, mode="nearest")(
            target.float()).expand(feat_S.size())
        tar_feat_T = nn.Upsample(size_f, mode="nearest")(
            target.float()).expand(feat_T.size())
        center_feat_S = feat_S.clone().detach()
        center_feat_T = feat_T.clone().detach()
        for i in range(self.num_classes):
            mask_feat_S = (tar_feat_S == i).float()
            mask_feat_T = (tar_feat_T == i).float()
            center_feat_S = (1 - mask_feat_S) * center_feat_S + mask_feat_S * (
                (mask_feat_S * feat_S).sum(dim=self.mean_dims, keepdim=True) /
                (mask_feat_S.sum(dim=self.mean_dims, keepdim=True) + 1e-6))
            center_feat_T = (1 - mask_feat_T) * center_feat_T + mask_feat_T * (
                (mask_feat_T * feat_T).sum(dim=self.mean_dims, keepdim=True) /
                (mask_feat_T.sum(dim=self.mean_dims, keepdim=True) + 1e-6))

        # cosinesimilarity along C
        cos = nn.CosineSimilarity(dim=1)
        pcsim_feat_S = cos(feat_S, center_feat_S)
        pcsim_feat_T = cos(feat_T, center_feat_T)

        # mseloss
        mse = nn.MSELoss(reduction="none")
        loss = mse(pcsim_feat_S, pcsim_feat_T)
        target_tmp = F.interpolate(target.float(),
                                   size=loss.shape[-2:],
                                   mode="nearest").squeeze(1)
        index = target_tmp != self.ignore_index
        if torch.any(index):
            return self.loss_weight * loss[index].mean()
        else:
            return loss.mean() * 0

    def extra_repr(self):
        s = f"lossweight={self.loss_weight}, class_nums={self.num_classes}, \
              mean_dims={self.mean_dims}"

        return s


def args_adaptor(np_args):
    logit = torch.from_numpy(np_args[0]).cuda()
    logit.requires_grad = True
    softlabel = torch.from_numpy(np_args[1]).cuda()
    label = torch.from_numpy(np_args[2]).cuda()
    label = torch.argmax(label, dim=1, keepdim=True)

    return [logit, softlabel, label]


def executer_creator():
    coder_instance = CriterionIFV(2).cuda()
    return Executer(coder_instance.forward, args_adaptor)
