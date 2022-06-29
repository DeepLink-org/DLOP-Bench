from long_tail_bench.core.executer import Executer
import torch
from .assign_result import AssignResult


def assign_wrt_overlaps(overlaps, assigned_gt_inds=None, gt_labels=None):
    """Assign w.r.t. the overlaps of bboxes with gts.

    Args:
        overlaps (Tensor): Overlaps between k gt_bboxes and n bboxes,
            shape(k, n).
        gt_labels (Tensor, optional): Labels of k gt_bboxes, shape (k, ).

    Returns:
        :obj:`AssignResult`: The assign result.
    """
    num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

    if num_gts == 0 or num_bboxes == 0:
        # No ground truth or boxes, return empty assignment
        max_overlaps = overlaps.new_zeros((num_bboxes, ))
        if num_gts == 0:
            # No truth, assign everything to background
            assigned_gt_inds[:] = 0
        if gt_labels is None:
            assigned_labels = None
        else:
            assigned_labels = overlaps.new_full((num_bboxes, ),
                                                -1,
                                                dtype=torch.long)
        return AssignResult(num_gts,
                            assigned_gt_inds,
                            max_overlaps,
                            labels=assigned_labels)

    # for each anchor, which gt best overlaps with it
    # for each anchor, the max iou of all gts
    max_overlaps, argmax_overlaps = overlaps.max(dim=0)
    # for each gt, which anchor best overlaps with it
    # for each gt, the max iou of all proposals
    gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

    # 2. assign negative: below
    # the negative inds are set to be 0
    neg_iou_thr = 0.9
    pos_iou_thr = 0.1
    min_pos_iou = 0.5
    match_low_quality = True
    # gt_max_assign_all = True
    if isinstance(neg_iou_thr, float):
        # assigned_gt_inds[(max_overlaps >= 0) & (max_overlaps < neg_iou_thr)] = 0  # noqa
        indexing = (max_overlaps >= 0) & (max_overlaps < neg_iou_thr)
        indexing = indexing.nonzero().transpose(0, 1)
        assigned_gt_inds[indexing] = 0
    elif isinstance(neg_iou_thr, tuple):
        assert len(neg_iou_thr) == 2
        assigned_gt_inds[(max_overlaps >= neg_iou_thr[0])
                         & (max_overlaps < neg_iou_thr[1])] = 0

    # 3. assign positive: above positive IoU threshold
    pos_inds = max_overlaps >= pos_iou_thr
    # assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1
    # parrost do not support boolean indexing yet, so we use nonzero, transpose
    # and split to replace it
    pos_indexing = pos_inds.nonzero().transpose(0, 1)
    assigned_gt_inds[pos_indexing] = argmax_overlaps[pos_indexing] + 1

    if match_low_quality:
        # Low-quality matching will overwirte the assigned_gt_inds assigned
        # in Step 3. Thus, the assigned gt might not be the best one for
        # prediction.
        # For example, if bbox A has 0.9 and 0.8 iou with GT bbox 1 & 2,
        # bbox 1 will be assigned as the best target for bbox A in step 3.
        # However, if GT bbox 2's gt_argmax_overlaps = A, bbox A's
        # assigned_gt_inds will be overwritten to be bbox B.
        # This might be the reason that it is not used in ROI Heads.
        # for i in range(num_gts):
        #     if gt_max_overlaps[i] >= min_pos_iou:
        #         if gt_max_assign_all:
        #             max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
        #             # assigned_gt_inds[max_iou_inds] = i + 1
        #             max_iou_indexing = max_iou_inds.nonzero().transpose(0, 1)
        #             assigned_gt_inds[max_iou_indexing] = i + 1
        #         else:
        #             assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1
        # The optimized code
        mask = gt_max_overlaps >= min_pos_iou
        inds = mask.nonzero().transpose(0, 1)
        assigned_gt_inds[gt_argmax_overlaps[inds]] = inds + 1

    if gt_labels is not None:
        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
        pos_inds = torch.nonzero(assigned_gt_inds > 0,
                                 as_tuple=False).squeeze()
        if pos_inds.numel() > 0:
            assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] -
                                                  1]
    else:
        assigned_labels = None

    return AssignResult(num_gts,
                        assigned_gt_inds,
                        max_overlaps,
                        labels=assigned_labels)


def args_adaptor(np_args):
    overlaps = torch.from_numpy(np_args[0]).cuda()
    # 1. assign -1 by default
    assigned_gt_inds = overlaps.new_full((overlaps.size(1), ),
                                         -1,
                                         dtype=torch.long)
    return [overlaps, assigned_gt_inds]


def executer_creator():
    return Executer(assign_wrt_overlaps,
                    args_adaptor).register_custom_class(AssignResult)
