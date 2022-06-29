import torch
import torch.nn as nn
import torch.nn.functional as F
from long_tail_bench.core.executer import Executer


class InterRegion(nn.Module):
    label_keys = ["feats_T", "gt_semantic_seg"]

    def __init__(self, classes, loss_weight=1.0):
        super(InterRegion, self).__init__()
        self.num_classes = classes
        self.loss_weight = loss_weight

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
        center_feat_S = torch.zeros((feat_S.shape[0], feat_S.shape[1],
                                     self.num_classes)).type_as(feat_S)
        center_feat_T = torch.zeros((feat_T.shape[0], feat_T.shape[1],
                                     self.num_classes)).type_as(feat_T)
        for i in range(self.num_classes):
            mask_feat_S = (tar_feat_S == i).float()
            mask_feat_T = (tar_feat_T == i).float()
            center_feat_S[:, :, i] = (mask_feat_S * feat_S).sum(-1).sum(-1) / (
                mask_feat_S.sum(-1).sum(-1) + 1e-6)
            center_feat_T[:, :, i] = (mask_feat_T * feat_T).sum(-1).sum(-1) / (
                mask_feat_T.sum(-1).sum(-1) + 1e-6)

        # cosinesimilarity along C
        center_feat_S = center_feat_S.unsqueeze(3).repeat(
            1, 1, 1, self.num_classes)
        center_feat_T = center_feat_T.unsqueeze(3).repeat(
            1, 1, 1, self.num_classes)
        cos = nn.CosineSimilarity(dim=1)
        pcsim_feat_S = cos(center_feat_S, center_feat_S.permute(0, 1, 3, 2))
        pcsim_feat_T = cos(center_feat_T, center_feat_T.permute(0, 1, 3, 2))

        # mseloss
        mse = nn.MSELoss()
        loss = mse(pcsim_feat_S, pcsim_feat_T)
        return self.loss_weight * loss

    def extra_repr(self):
        s = f"lossweight={self.loss_weight}, class_nums={self.num_classes}"
        return s


def args_adaptor(np_args):
    logit = torch.from_numpy(np_args[0]).cuda()
    logit.requires_grad = True

    softlabel = torch.from_numpy(np_args[1]).cuda()

    label = torch.from_numpy(np_args[2]).cuda()
    label = torch.argmax(label, dim=1, keepdim=True)

    return [logit, softlabel, label]


def executer_creator():
    coder_instance = InterRegion(2).cuda()
    return Executer(coder_instance.forward, args_adaptor)
