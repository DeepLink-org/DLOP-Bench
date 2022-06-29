import torch
from long_tail_bench.core.executer import Executer


def yolo_decode(bboxes, pred_bboxes, stride):
    """Apply transformation `pred_bboxes` to `boxes`.

    Args:
        boxes (torch.Tensor): Basic boxes, e.g. anchors.
        pred_bboxes (torch.Tensor): Encoded boxes with shape
        stride (torch.Tensor | int): Strides of bboxes.

    Returns:
        torch.Tensor: Decoded boxes.
    """
    assert pred_bboxes.size(0) == bboxes.size(0)
    assert pred_bboxes.size(-1) == bboxes.size(-1) == 4
    x_center = (bboxes[..., 0] + bboxes[..., 2]) * 0.5
    y_center = (bboxes[..., 1] + bboxes[..., 3]) * 0.5
    w = bboxes[..., 2] - bboxes[..., 0]
    h = bboxes[..., 3] - bboxes[..., 1]
    # Get outputs x, y
    x_center_pred = (pred_bboxes[..., 0] - 0.5) * stride + x_center
    y_center_pred = (pred_bboxes[..., 1] - 0.5) * stride + y_center
    w_pred = torch.exp(pred_bboxes[..., 2]) * w
    h_pred = torch.exp(pred_bboxes[..., 3]) * h

    decoded_bboxes = torch.stack(
        (x_center_pred - w_pred / 2, y_center_pred - h_pred / 2,
         x_center_pred + w_pred / 2, y_center_pred + h_pred / 2),
        dim=-1)

    return decoded_bboxes


def args_adaptor(np_args):
    bboxes = torch.from_numpy(np_args[0]).cuda()
    gt_bboxes = torch.from_numpy(np_args[1]).cuda()
    stride = torch.from_numpy(np_args[2]).cuda()

    return [bboxes, gt_bboxes, stride]


def executer_creator():
    return Executer(yolo_decode, args_adaptor)
