import numpy as np
import torch
import torch.nn.functional as F
from long_tail_bench.core.executer import Executer


def bbox_rescale(bboxes, scale_factor=1.0):
    """Rescale bounding box w.r.t. scale_factor.

    Args:
        bboxes (Tensor): Shape (n, 4) for bboxes or (n, 5) for rois
        scale_factor (float): rescale factor

    Returns:
        Tensor: Rescaled bboxes.
    """
    if bboxes.size(1) == 5:
        bboxes_ = bboxes[:, 1:]
        inds_ = bboxes[:, 0]
    else:
        bboxes_ = bboxes
    cx = (bboxes_[:, 0] + bboxes_[:, 2]) * 0.5
    cy = (bboxes_[:, 1] + bboxes_[:, 3]) * 0.5
    w = bboxes_[:, 2] - bboxes_[:, 0]
    h = bboxes_[:, 3] - bboxes_[:, 1]
    w = w * scale_factor
    h = h * scale_factor
    x1 = cx - 0.5 * w
    x2 = cx + 0.5 * w
    y1 = cy - 0.5 * h
    y2 = cy + 0.5 * h
    if bboxes.size(1) == 5:
        rescaled_bboxes = torch.stack([inds_, x1, y1, x2, y2], dim=-1)
    else:
        rescaled_bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
    return rescaled_bboxes


def bucket2bbox(
    proposals,
    cls_preds,
    offset_preds,
    num_buckets,
    scale_factor=1.0,
    max_shape=None,
    clip_border=True,
):
    """Apply bucketing estimation (cls preds) and fine regression (offset
    preds) to generate det bboxes.

    Args:
        proposals (Tensor): Boxes to be transformed. Shape (n, 4)
        cls_preds (Tensor): bucketing estimation. Shape (n, num_buckets*2).
        offset_preds (Tensor): fine regression. Shape (n, num_buckets*2).
        num_buckets (int): Number of buckets.
        scale_factor (float): Scale factor to rescale proposals.
        max_shape (tuple[int, int]): Maximum bounds for boxes. specifies (H, W)
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.

    Returns:
        tuple[Tensor]: (bboxes, loc_confidence).

            - bboxes: predicted bboxes. Shape (n, 4)
            - loc_confidence: localization confidence of predicted bboxes.
                Shape (n,).
    """

    side_num = int(np.ceil(num_buckets / 2.0))
    cls_preds = cls_preds.view(-1, side_num)
    offset_preds = offset_preds.view(-1, side_num)

    scores = F.softmax(cls_preds, dim=1)
    score_topk, score_label = scores.topk(2, dim=1, largest=True, sorted=True)

    rescaled_proposals = bbox_rescale(proposals, scale_factor)

    pw = rescaled_proposals[..., 2] - rescaled_proposals[..., 0]
    ph = rescaled_proposals[..., 3] - rescaled_proposals[..., 1]
    px1 = rescaled_proposals[..., 0]
    py1 = rescaled_proposals[..., 1]
    px2 = rescaled_proposals[..., 2]
    py2 = rescaled_proposals[..., 3]

    bucket_w = pw / num_buckets
    bucket_h = ph / num_buckets

    score_inds_l = score_label[0::4, 0]
    score_inds_r = score_label[1::4, 0]
    score_inds_t = score_label[2::4, 0]
    score_inds_d = score_label[3::4, 0]
    l_buckets = px1 + (0.5 + score_inds_l.float()) * bucket_w
    r_buckets = px2 - (0.5 + score_inds_r.float()) * bucket_w
    t_buckets = py1 + (0.5 + score_inds_t.float()) * bucket_h
    d_buckets = py2 - (0.5 + score_inds_d.float()) * bucket_h

    offsets = offset_preds.view(-1, 4, side_num)
    inds = torch.arange(proposals.size(0)).to(proposals).long()
    # NOTE(limaolin): This is a value dependent case, inds is a Tensor
    l_offsets = offsets[:, 0, :][inds, score_inds_l]
    r_offsets = offsets[:, 1, :][inds, score_inds_r]
    t_offsets = offsets[:, 2, :][inds, score_inds_t]
    d_offsets = offsets[:, 3, :][inds, score_inds_d]

    x1 = l_buckets - l_offsets * bucket_w
    x2 = r_buckets - r_offsets * bucket_w
    y1 = t_buckets - t_offsets * bucket_h
    y2 = d_buckets - d_offsets * bucket_h

    if clip_border and max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1] - 1)
        y1 = y1.clamp(min=0, max=max_shape[0] - 1)
        x2 = x2.clamp(min=0, max=max_shape[1] - 1)
        y2 = y2.clamp(min=0, max=max_shape[0] - 1)
    bboxes = torch.cat([x1[:, None], y1[:, None], x2[:, None], y2[:, None]],
                       dim=-1)

    # bucketing guided rescoring
    loc_confidence = score_topk[:, 0]
    top2_neighbor_inds = (score_label[:, 0] - score_label[:, 1]).abs() == 1
    loc_confidence += score_topk[:, 1] * top2_neighbor_inds.float()
    loc_confidence = loc_confidence.view(-1, 4).mean(dim=1)

    return bboxes, loc_confidence


def args_adaptor(np_args):
    proposals = torch.from_numpy(np_args[0]).cuda()
    cls_preds = torch.from_numpy(np_args[1]).cuda()
    offset_preds = torch.from_numpy(np_args[2]).cuda()
    num_buckets = np_args[3]
    scale_factor = np_args[4]

    return [proposals, cls_preds, offset_preds, num_buckets, scale_factor]


def executer_creator():
    return Executer(bucket2bbox, args_adaptor)
