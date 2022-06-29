import torch
from long_tail_bench.core.executer import Executer


def l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    assert pred.size() == target.size() and target.numel() > 0
    loss = torch.abs(pred - target)
    return loss


def carl_loss(
    cls_score,
    labels,
    bbox_pred,
    bbox_targets,
    loss_bbox,
    k=1,
    bias=0.2,
    avg_factor=None,
    sigmoid=False,
    num_class=80,
):
    """Classification-Aware Regression Loss (CARL).

    Args:
        cls_score (Tensor): Predicted classification scores.
        labels (Tensor): Targets of classification.
        bbox_pred (Tensor): Predicted bbox deltas.
        bbox_targets (Tensor): Target of bbox regression.
        loss_bbox (func): Regression loss func of the head.
        bbox_coder (obj): BBox coder of the head.
        k (float): Power of the non-linear mapping.
        bias (float): Shift of the non-linear mapping.
        avg_factor (int): Average factor used in regression loss.
        sigmoid (bool): Activation of the classification score.
        num_class (int): Number of classes, default: 80.

    Return:
        dict: CARL loss dict.
    """
    pos_label_inds = (((labels >= 0) &
                       (labels < num_class)).nonzero().reshape(-1))

    if pos_label_inds.numel() == 0:
        return dict(loss_carl=cls_score.sum()[None] * 0.0)
    pos_labels = labels[pos_label_inds]

    # multiply pos_cls_score with the corresponding bbox weight
    # and remain gradient
    if sigmoid:
        pos_cls_score = cls_score.sigmoid()[pos_label_inds, pos_labels]
    else:
        pos_cls_score = cls_score.softmax(-1)[pos_label_inds, pos_labels]
    carl_loss_weights = (bias + (1 - bias) * pos_cls_score).pow(k)

    # normalize carl_loss_weight to make its sum equal to num positive
    num_pos = float(pos_cls_score.size(0))
    weight_ratio = num_pos / carl_loss_weights.sum()
    carl_loss_weights *= weight_ratio

    if avg_factor is None:
        avg_factor = bbox_targets.size(0)
    # if is class agnostic, bbox pred is in shape (N, 4)
    # otherwise, bbox pred is in shape (N, #classes, 4)
    if bbox_pred.size(-1) > 4:
        bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 4)
        pos_bbox_preds = bbox_pred[pos_label_inds, pos_labels]
    else:
        pos_bbox_preds = bbox_pred[pos_label_inds]
    ori_loss_reg = (loss_bbox(pos_bbox_preds, bbox_targets[pos_label_inds]) /
                    avg_factor)
    loss_carl = (ori_loss_reg * carl_loss_weights[:, None]).sum()
    return loss_carl


def args_adaptor(np_args):
    cls_score = torch.from_numpy(np_args[0]).cuda()
    labels = torch.from_numpy(np_args[1]).cuda()
    bbox_pred = torch.from_numpy(np_args[2]).cuda()
    bbox_targets = torch.from_numpy(np_args[3]).cuda()
    loss_bbox = l1_loss

    return [cls_score, labels, bbox_pred, bbox_targets, loss_bbox]


def executer_creator():
    return Executer(carl_loss, args_adaptor)
