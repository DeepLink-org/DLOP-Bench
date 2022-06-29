import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch.onnx

import torch.nn as nn
import torch.nn.init as init
from gen_data import gen_np_args, args_adaptor

import onnx

def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction="mean", avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == "mean":
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != "none":
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def varifocal_loss(
    pred,
    target,
    weight=None,
    alpha=0.75,
    gamma=2.0,
    iou_weighted=True,
    reduction="mean",
    avg_factor=None,
):
    """`Varifocal Loss <https://arxiv.org/abs/2008.13367>`_

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning target of the iou-aware
            classification score with shape (N, C), C is the number of classes.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction. Defaults to None.
        alpha (float, optional): A balance factor for the negative part of
            Varifocal Loss, which is different from the alpha of Focal Loss.
            Defaults to 0.75.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        iou_weighted (bool, optional): Whether to weight the loss of the
            positive example with the iou target. Defaults to True.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    # pred and target should be of the same size
    assert pred.size() == target.size()
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    if iou_weighted:
        focal_weight = (target * (target > 0.0).float() + alpha *
                        (pred_sigmoid - target).abs().pow(gamma) *
                        (target <= 0.0).float())
    else:
        focal_weight = (target > 0.0).float() + alpha * (
            pred_sigmoid - target).abs().pow(gamma) * (target <= 0.0).float()
    loss = (
        F.binary_cross_entropy_with_logits(pred, target, reduction="none") *
        focal_weight)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


class Bbox(nn.Module):
    def __init__(self):
        super(Bbox, self).__init__()
        self.varifocal_loss = varifocal_loss

    def forward(self, pred, target):
        loss = self.varifocal_loss(pred, target)

        return loss 

torch_model = Bbox()

torch_model.eval()

pred, target = args_adaptor(gen_np_args(128, 4))
torch_out = torch_model(pred, target)

torch.onnx.export(torch_model, 
        (pred, target),
        "varifocal_loss.onnx",
        verbose=True,
        export_params=True,
        opset_version=12,
        input_names=['pred', 'target'],
        output_names = ['output'])
