import torch
import torch.nn as nn
from bench.core.executer import Executer


class CenterNetKeyPoint(nn.Module):
    def parse_preds(self, mlvl_preds):
        """permute mlvl_preds for loss calculation"""
        preds = [list(t) for t in zip(*mlvl_preds)]
        return preds[:3], preds[3:5], preds[5:]

    def get_losses(self, mlvl_preds, targets):
        heat_maps, embeddings, kp_offsets = self.parse_preds(mlvl_preds)
        gt_heat_maps, gt_masks, gt_kp_offsets = targets

        num_lvl = len(heat_maps[0])
        # focal loss
        """ gaussian_sigmoid_focal_loss: ADDED
        focal_loss = 0
        for maps, gt_maps in zip(heat_maps, gt_heat_maps):
            for m, gt_m in zip(maps, gt_maps):
                focal_loss += self.cls_loss(m, gt_m)
        """
        # corner grouping loss
        """ pull & push loss: ADDED
        pull_loss = 0
        push_loss = 0
        for tl_emb, br_emb, gt_mask in zip(*embeddings, *gt_masks):
            pull = self.pull_loss([tl_emb, br_emb], gt_mask)
            push = self.push_loss([tl_emb, br_emb], gt_mask)
            pull_loss += pull
            push_loss += push
        """
        # key point regression loss
        regr_loss = 0
        for kps, gt_kps in zip(kp_offsets, gt_kp_offsets):
            for kp, gt_kp, gt_mask in zip(kps, gt_kps, *gt_masks):
                num = gt_mask.float().sum() + 1e-4
                temp = gt_mask.unsqueeze(2)
                print(
                    "gt_kp shape:",
                    gt_kp.shape,
                    "gt_mask shape:",
                    gt_mask.shape,
                    "gt_mask unsqueeze shape:",
                    temp.shape,
                )
                gt_mask = temp.expand_as(gt_kp)
                kp = kp[gt_mask].float()
                gt_kp = gt_kp[gt_mask]
                """ smooth_l1_loss: ADDED
                regr_loss += self.regr_loss(kp, gt_kp, 'mean', num)
                """
                regr_loss += torch.tensor([kp + gt_kp + num])

        return regr_loss / num_lvl


def args_adaptor(np_args):
    heatmap = torch.from_numpy(np_args[0]).cuda()
    heatmap.requires_grad_(True)
    tl_heat = ct_heat = br_heat = heatmap
    embedding = torch.from_numpy(np_args[1]).cuda()
    embedding.requires_grad_(True)
    tl_emb = ct_emb = br_emb = embedding
    offset = torch.from_numpy(np_args[2]).cuda()
    offset.requires_grad_(True)
    tl_offset = ct_offset = br_offset = offset
    mlvl_preds = [[
        tl_heat,
        ct_heat,
        br_heat,
        tl_emb,
        ct_emb,
        br_emb,
        tl_offset,
        ct_offset,
        br_offset,
    ]] * 2

    N = np_args[6]
    heatmap = torch.from_numpy(np_args[3]).cuda()
    tl_heat = ct_heat = br_heat = heatmap
    embedding = torch.from_numpy(np_args[4]).cuda()
    tl_emb = ct_emb = br_emb = embedding
    offset = torch.from_numpy(np_args[5]).cuda()
    tl_offset = ct_offset = br_offset = offset

    target = [
        [tl_heat, ct_heat, br_heat],
        [
            [embedding] * N,
        ],
        [tl_offset, ct_offset, br_offset],
    ]

    return [mlvl_preds, target]


def executer_creator():
    coder_instance = CenterNetKeyPoint()
    return Executer(coder_instance.get_losses, args_adaptor)
