# Copyright(c) OpenMMLab. All Rights Reserved.
# Copied from
import torch
from bench.core.executer import Executer

GPU_MEMORY = None


def vanilla_bbox_iof_overlaps(b1, b2):
    """
    Arguments:
        b1: dts, [n, >=4] (x1, y1, x2, y2, ...)
        b1: gts, [n, >=4] (x1, y1, x2, y2, ...)

    Returns:
        intersection-over-former-box pair-wise
    """
    area1 = (b1[:, 2] - b1[:, 0] + 1) * (b1[:, 3] - b1[:, 1] + 1)
    lt = torch.max(b1[:, None, :2], b2[:, :2])
    rb = torch.min(b1[:, None, 2:4], b2[:, 2:4])
    wh = (rb - lt + 1).clamp(min=0)
    inter_area = wh[:, :, 0] * wh[:, :, 1]
    return inter_area / torch.clamp(area1[:, None], min=1)


def gpu_iou_overlap(b1, b2, mode="IoU"):
    """compute IoU/IoF/IoS between b1 and b2
    Args:
        b1: Tensor, [N, >=4]
        b2: Tensor, [M, >=4]
        mode: str, {IoU, IoF, IoS} Intersection over Union/First/Second area
    """
    if b1.numel() == 0 or b2.numel() == 0:
        return b1.new_zeros((0, ))

    # flag = {"IoU": 0, "IoF": 1, "IoS": 2}[mode]

    assert b1.shape[1] >= 4 and b2.shape[1] >= 4
    assert b1.is_cuda and b2.is_cuda

    b1 = b1[:, :4].contiguous()
    b2 = b2[:, :4].contiguous()
    ious = b1.new_zeros((b1.shape[0], b2.shape[0]))
    # overlaps.iou(b1, b2, ious, flag)
    return ious


def vanilla_bbox_iou_overlaps(b1, b2):
    """
    Arguments:
        b1: dts, [n, >=4] (x1, y1, x2, y2, ...)
        b1: gts, [n, >=4] (x1, y1, x2, y2, ...)

    Returns:
        intersection-over-union pair-wise.
    """
    area1 = (b1[:, 2] - b1[:, 0] + 1) * (b1[:, 3] - b1[:, 1] + 1)
    area2 = (b2[:, 2] - b2[:, 0] + 1) * (b2[:, 3] - b2[:, 1] + 1)
    lt = torch.max(b1[:, None, :2], b2[:, :2])
    rb = torch.min(b1[:, None, 2:4], b2[:, 2:4])
    wh = (rb - lt + 1).clamp(min=0)
    inter_area = wh[:, :, 0] * wh[:, :, 1]
    union_area = area1[:, None] + area2 - inter_area
    return inter_area / torch.clamp(union_area, min=1)


def bbox_iou_overlaps(b1, b2):
    if not b1.is_cuda:
        return vanilla_bbox_iou_overlaps(b1, b2)

    global GPU_MEMORY
    gbytes = 1024.0**3
    if GPU_MEMORY is None:
        GPU_MEMORY = torch.cuda.get_device_properties(
            b1.device.index).total_memory
    alloated_memory = torch.cuda.memory_allocated()
    spare_memory = 0.5 * gbytes
    available_memory = GPU_MEMORY - alloated_memory - spare_memory
    size = b1.shape[0] * b2.shape[0]
    needed_memory = 2 * size * 4

    if needed_memory < available_memory:
        ious = gpu_iou_overlap(b1, b2, mode="IoU")
    else:
        ious = vanilla_bbox_iou_overlaps(b1.cpu(), b2.cpu())
        res_memory = size * 4
        if res_memory < available_memory:
            ious = ious.to(b1.device)
    return ious


def bbox_iof_overlaps(b1, b2):
    if not b1.is_cuda:
        return vanilla_bbox_iof_overlaps(b1, b2)
    global GPU_MEMORY
    gbytes = 1024.0**3
    if GPU_MEMORY is None:
        GPU_MEMORY = torch.cuda.get_device_properties(
            b1.device.index).total_memory
    alloated_memory = torch.cuda.memory_allocated()
    spare_memory = 0.5 * gbytes
    available_memory = GPU_MEMORY - alloated_memory - spare_memory
    size = b1.shape[0] * b2.shape[0]
    needed_memory = 2 * size * 4

    if needed_memory < available_memory:
        ious = gpu_iou_overlap(b1, b2, mode="IoF")
    else:
        ious = vanilla_bbox_iof_overlaps(b1.cpu(), b2.cpu())
        res_memory = size * 4
        if res_memory < available_memory:
            ious = ious.to(b1.device)
    return ious


def cuda_memory_available(device, needed_memory):
    if torch.cuda.is_available():
        global GPU_MEMORY
        gbytes = 1024.0**3
        if GPU_MEMORY is None:
            GPU_MEMORY = torch.cuda.get_device_properties(
                device.index).total_memory
        alloated_memory = torch.cuda.memory_allocated()
        spare_memory = 0.5 * gbytes
        available_memory = GPU_MEMORY - alloated_memory - spare_memory
        if needed_memory < available_memory:
            return True
    return False


class MaxIoUMatcher(object):
    NEGATIVE_TARGET = -1
    IGNORE_TARGET = -2

    def __init__(
        self,
        negative_iou_thresh,
        positive_iou_thresh,
        ignore_iou_thresh,
        allow_low_quality_match,
        low_quality_thresh=0,
    ):

        self.negative_iou_thresh = negative_iou_thresh
        self.positive_iou_thresh = positive_iou_thresh
        self.ignore_iou_thresh = ignore_iou_thresh
        self.allow_low_quality_match = allow_low_quality_match
        self.low_quality_thresh = low_quality_thresh

    def match(
        self,
        candidate_boxes,
        gt_bboxes,
        gt_ignores=None,
        return_max_overlaps=False,
    ):
        """
        Match roi to gt

        .. note::

            This function involves some temporarily tensors.

            - overlaps (``FloatTensor``): [N, M], ious between bounding boxes
            [N, 4] and gt boxes (M, 4), boxes format is (x1, y1, x2, y2)
            - ignore_overlaps (``FloatTensor``): [N, K], ious between bounding
            boxes [N, 4] with ignore regions [M, 4]

        Arguments:
            - candidate_boxes (``FloatTensor``): [N, 4] (x1, y1, x2, y2)
            - gt_bboxes (``FloatTensor``): [M, 5] (x1, y1, x2, y2, label)
            - gt_ignores (``FloatTensor`` or ``None``): [G, 4] (x1, y1, x2, y2)

        Returns:
            - target (LongTensor): [N], matched gt index for each RoI.

        .. note::

            1. if a roi is positive, target = matched gt index (>=0)
            2. if a roi is negative, target = -1,
            3. if a roi is ignored,  target = -2;
        """
        device = candidate_boxes.device
        # we use cpu tensor if there are too many bboxes, gts and ignores
        N, M = candidate_boxes.shape[0], gt_bboxes.shape[0]
        if gt_ignores is not None:
            M += gt_ignores.shape[0]
        if not cuda_memory_available(candidate_boxes.device,
                                     N * M * 2 * 4 * 4):
            candidate_boxes = candidate_boxes.cpu()
            gt_bboxes = gt_bboxes.cpu()
            if gt_ignores is not None:
                gt_ignores = gt_ignores.cpu()

        N = candidate_boxes.shape[0]
        M = gt_bboxes.shape[0]

        # check M > 0 for no-gt support
        if M == 0:
            overlaps = candidate_boxes.new_zeros(N, 1)
        else:
            overlaps = bbox_iou_overlaps(candidate_boxes, gt_bboxes)

        target = candidate_boxes.new_full((N, ),
                                          self.IGNORE_TARGET,
                                          dtype=torch.int64)
        dt_to_gt_max, dt_to_gt_argmax = overlaps.max(dim=1)

        # rule 1: negative if maxiou < negative_iou_thresh:
        neg_mask = dt_to_gt_max < self.negative_iou_thresh
        target[neg_mask] = self.NEGATIVE_TARGET

        # rule 2: positive if maxiou > pos_iou_thresh
        pos_mask = dt_to_gt_max > self.positive_iou_thresh
        target[pos_mask] = dt_to_gt_argmax[pos_mask]

        # rule 3: positive if a dt has highest iou with any gt
        if self.allow_low_quality_match and M > 0:
            overlaps = overlaps.t()  # IMPORTANT, for faster caculation
            gt_to_dt_max, _ = overlaps.max(dim=1)
            temp_mask = overlaps >= gt_to_dt_max[:, None] - 1e-3
            lqm_dt_inds = torch.nonzero(temp_mask.any(dim=0)).reshape(-1)
            if lqm_dt_inds.numel() > 0:
                target[lqm_dt_inds] = dt_to_gt_argmax[lqm_dt_inds]
                pos_mask[lqm_dt_inds] = 1

        del overlaps
        ignore_overlaps = None
        if gt_ignores is not None and gt_ignores.numel() > 0:
            ignore_overlaps = bbox_iof_overlaps(candidate_boxes, gt_ignores)

        # rule 4: dt has high iou with ignore regions may not supposed to be
        # negative
        if ignore_overlaps is not None and ignore_overlaps.numel() > 0:
            dt_to_ig_max, _ = ignore_overlaps.max(dim=1)
            ignored_dt_mask = dt_to_ig_max > self.ignore_iou_thresh
            # remove positives from ignored
            ignored_dt_mask = ignored_dt_mask ^ (ignored_dt_mask & pos_mask)
            target[ignored_dt_mask] = self.IGNORE_TARGET

        if return_max_overlaps:
            return target.to(device), dt_to_gt_max
        else:
            return target.to(device)


def args_generator():
    raise NotImplementedError()


def executer_creator():
    coder_instance = MaxIoUMatcher(0.3, 0.3, 0.4, True)
    return Executer(coder_instance.match, args_generator)
