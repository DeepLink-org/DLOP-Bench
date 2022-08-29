import torch
import torch.nn.functional as F
import numpy as np
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


def generat_buckets(proposals, num_buckets, scale_factor=1.0):
    """Generate buckets w.r.t bucket number and scale factor of proposals.

    Args:
        proposals (Tensor): Shape (n, 4)
        num_buckets (int): Number of buckets.
        scale_factor (float): Scale factor to rescale proposals.

    Returns:
        tuple[Tensor]: (bucket_w, bucket_h, l_buckets, r_buckets,
         t_buckets, d_buckets)

            - bucket_w: Width of buckets on x-axis. Shape (n, ).
            - bucket_h: Height of buckets on y-axis. Shape (n, ).
            - l_buckets: Left buckets. Shape (n, ceil(side_num/2)).
            - r_buckets: Right buckets. Shape (n, ceil(side_num/2)).
            - t_buckets: Top buckets. Shape (n, ceil(side_num/2)).
            - d_buckets: Down buckets. Shape (n, ceil(side_num/2)).
    """
    proposals = bbox_rescale(proposals, scale_factor)

    # number of buckets in each side
    side_num = int(np.ceil(num_buckets / 2.0))
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]
    px1 = proposals[..., 0]
    py1 = proposals[..., 1]
    px2 = proposals[..., 2]
    py2 = proposals[..., 3]

    bucket_w = pw / num_buckets
    bucket_h = ph / num_buckets

    # left buckets
    l_buckets = (
        px1[:, None] +
        (0.5 + torch.arange(0, side_num).to(proposals).float())[None, :] *
        bucket_w[:, None])
    # right buckets
    r_buckets = (
        px2[:, None] -
        (0.5 + torch.arange(0, side_num).to(proposals).float())[None, :] *
        bucket_w[:, None])
    # top buckets
    t_buckets = (
        py1[:, None] +
        (0.5 + torch.arange(0, side_num).to(proposals).float())[None, :] *
        bucket_h[:, None])
    # down buckets
    d_buckets = (
        py2[:, None] -
        (0.5 + torch.arange(0, side_num).to(proposals).float())[None, :] *
        bucket_h[:, None])
    return bucket_w, bucket_h, l_buckets, r_buckets, t_buckets, d_buckets


def bbox2bucket(
    proposals,
    gt,
    num_buckets,
    scale_factor,
    offset_topk=2,
    offset_upperbound=1.0,
    cls_ignore_neighbor=True,
):
    """Generate buckets estimation and fine regression targets.

    Args:
        proposals (Tensor): Shape (n, 4)
        gt (Tensor): Shape (n, 4)
        num_buckets (int): Number of buckets.
        scale_factor (float): Scale factor to rescale proposals.
        offset_topk (int): Topk buckets are used to generate
             bucket fine regression targets. Defaults to 2.
        offset_upperbound (float): Offset allowance to generate
             bucket fine regression targets.
             To avoid too large offset displacements. Defaults to 1.0.
        cls_ignore_neighbor (bool): Ignore second nearest bucket or Not.
             Defaults to True.

    Returns:
        tuple[Tensor]: (offsets, offsets_weights, bucket_labels, cls_weights).

            - offsets: Fine regression targets. \
                Shape (n, num_buckets*2).
            - offsets_weights: Fine regression weights. \
                Shape (n, num_buckets*2).
            - bucket_labels: Bucketing estimation labels. \
                Shape (n, num_buckets*2).
            - cls_weights: Bucketing estimation weights. \
                Shape (n, num_buckets*2).
    """
    assert proposals.size() == gt.size()

    # generate buckets
    proposals = proposals.float()
    gt = gt.float()
    (
        bucket_w,
        bucket_h,
        l_buckets,
        r_buckets,
        t_buckets,
        d_buckets,
    ) = generat_buckets(proposals, num_buckets, scale_factor)

    gx1 = gt[..., 0]
    gy1 = gt[..., 1]
    gx2 = gt[..., 2]
    gy2 = gt[..., 3]

    # generate offset targets and weights
    # offsets from buckets to gts
    l_offsets = (l_buckets - gx1[:, None]) / bucket_w[:, None]
    r_offsets = (r_buckets - gx2[:, None]) / bucket_w[:, None]
    t_offsets = (t_buckets - gy1[:, None]) / bucket_h[:, None]
    d_offsets = (d_buckets - gy2[:, None]) / bucket_h[:, None]

    # select top-k nearset buckets
    l_topk, l_label = l_offsets.abs().topk(offset_topk,
                                           dim=1,
                                           largest=False,
                                           sorted=True)
    r_topk, r_label = r_offsets.abs().topk(offset_topk,
                                           dim=1,
                                           largest=False,
                                           sorted=True)
    t_topk, t_label = t_offsets.abs().topk(offset_topk,
                                           dim=1,
                                           largest=False,
                                           sorted=True)
    d_topk, d_label = d_offsets.abs().topk(offset_topk,
                                           dim=1,
                                           largest=False,
                                           sorted=True)

    offset_l_weights = l_offsets.new_zeros(l_offsets.size())
    offset_r_weights = r_offsets.new_zeros(r_offsets.size())
    offset_t_weights = t_offsets.new_zeros(t_offsets.size())
    offset_d_weights = d_offsets.new_zeros(d_offsets.size())
    inds = torch.arange(0, proposals.size(0)).to(proposals).long()

    # generate offset weights of top-k nearset buckets
    for k in range(offset_topk):
        if k >= 1:
            # NOTE(limaolin): inds is a Tensor here, value dependent
            offset_l_weights[inds, l_label[:,
                                           k]] = (l_topk[:, k] <
                                                  offset_upperbound).float()
            offset_r_weights[inds, r_label[:,
                                           k]] = (r_topk[:, k] <
                                                  offset_upperbound).float()
            offset_t_weights[inds, t_label[:,
                                           k]] = (t_topk[:, k] <
                                                  offset_upperbound).float()
            offset_d_weights[inds, d_label[:,
                                           k]] = (d_topk[:, k] <
                                                  offset_upperbound).float()
        else:
            offset_l_weights[inds, l_label[:, k]] = 1.0
            offset_r_weights[inds, r_label[:, k]] = 1.0
            offset_t_weights[inds, t_label[:, k]] = 1.0
            offset_d_weights[inds, d_label[:, k]] = 1.0

    offsets = torch.cat([l_offsets, r_offsets, t_offsets, d_offsets], dim=-1)
    offsets_weights = torch.cat(
        [
            offset_l_weights,
            offset_r_weights,
            offset_t_weights,
            offset_d_weights,
        ],
        dim=-1,
    )

    # generate bucket labels and weight
    side_num = int(np.ceil(num_buckets / 2.0))
    labels = torch.stack(
        [l_label[:, 0], r_label[:, 0], t_label[:, 0], d_label[:, 0]], dim=-1)

    batch_size = labels.size(0)
    # NOTE(limaolin): one_hot is not traceable
    bucket_labels = (F.one_hot(labels.view(-1), side_num).view(batch_size,
                                                               -1).float())
    bucket_cls_l_weights = (l_offsets.abs() < 1).float()
    bucket_cls_r_weights = (r_offsets.abs() < 1).float()
    bucket_cls_t_weights = (t_offsets.abs() < 1).float()
    bucket_cls_d_weights = (d_offsets.abs() < 1).float()
    bucket_cls_weights = torch.cat(
        [
            bucket_cls_l_weights,
            bucket_cls_r_weights,
            bucket_cls_t_weights,
            bucket_cls_d_weights,
        ],
        dim=-1,
    )
    # ignore second nearest buckets for cls if necessay
    if cls_ignore_neighbor:
        bucket_cls_weights = (~((bucket_cls_weights == 1) &
                                (bucket_labels == 0))).float()
    else:
        bucket_cls_weights[:] = 1.0
    return offsets, offsets_weights, bucket_labels, bucket_cls_weights


def args_adaptor(np_args):
    proposals = torch.from_numpy(np_args[0]).cuda()
    gt = torch.from_numpy(np_args[1]).cuda()
    num_buckets = 8
    scale_factor = 1.0
    return [proposals, gt, num_buckets, scale_factor]


def executer_creator():
    return Executer(bbox2bucket, args_adaptor)
