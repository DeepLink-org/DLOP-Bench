import torch
from long_tail_bench.core.executer import Executer


def yolo_encode(bboxes, gt_bboxes, stride, eps: float):
    """Get box regression transformation deltas that can be used to
    transform the ``bboxes`` into the ``gt_bboxes``.

    Args:
        bboxes (torch.Tensor): Source boxes, e.g., anchors.
        gt_bboxes (torch.Tensor): Target of the transformation, e.g.,
            ground-truth boxes.
        stride (torch.Tensor | int): Stride of bboxes.

    Returns:
        torch.Tensor: Box transformation deltas
    """
    assert bboxes.size(0) == gt_bboxes.size(0)
    assert bboxes.size(-1) == gt_bboxes.size(-1) == 4
    x_center_gt = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) * 0.5
    y_center_gt = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) * 0.5
    w_gt = gt_bboxes[..., 2] - gt_bboxes[..., 0]
    h_gt = gt_bboxes[..., 3] - gt_bboxes[..., 1]
    x_center = (bboxes[..., 0] + bboxes[..., 2]) * 0.5
    y_center = (bboxes[..., 1] + bboxes[..., 3]) * 0.5
    w = bboxes[..., 2] - bboxes[..., 0]
    h = bboxes[..., 3] - bboxes[..., 1]
    w_target = torch.log((w_gt / w).clamp(min=eps))
    h_target = torch.log((h_gt / h).clamp(min=eps))
    x_center_target = ((x_center_gt - x_center) / stride + 0.5).clamp(
        eps, 1 - eps)
    y_center_target = ((y_center_gt - y_center) / stride + 0.5).clamp(
        eps, 1 - eps)
    encoded_bboxes = torch.stack(
        [x_center_target, y_center_target, w_target, h_target], dim=-1)
    return encoded_bboxes


def args_adaptor(np_args):
    bboxes = torch.from_numpy(np_args[0]).cuda()
    gt_bboxes = torch.from_numpy(np_args[1]).cuda()
    stride = torch.from_numpy(np_args[2]).cuda()
    eps = 1e-6

    return [bboxes, gt_bboxes, stride, eps]


def executer_creator():
    return Executer(yolo_encode, args_adaptor)
