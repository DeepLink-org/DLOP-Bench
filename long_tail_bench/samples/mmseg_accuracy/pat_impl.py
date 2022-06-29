import torch
from long_tail_bench.core.executer import Executer


def accuracy(pred, target, topk=1, thresh=None):
    """Calculate accuracy according to the prediction and target.

    Args:
        pred (torch.Tensor): The model prediction, shape (N, num_class, ...)
        target (torch.Tensor): The target of each prediction, shape (N, , ...)
        topk (int | tuple[int], optional): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thresh (float, optional): If not None, predictions with scores under
            this threshold are considered incorrect. Default to None.

    Returns:
        float | tuple[float]: If the input ``topk`` is a single integer,
            the function will return a single float as accuracy. If
            ``topk`` is a tuple containing multiple integers, the
            function will return a tuple containing accuracies of
            each ``topk`` number.
    """
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    if pred.size(0) == 0:
        accu = [pred.new_tensor(0.0) for i in range(len(topk))]
        return accu[0] if return_single else accu
    assert pred.ndim == target.ndim + 1
    assert pred.size(0) == target.size(0)
    assert maxk <= pred.size(
        1), f"maxk {maxk} exceeds pred dimension {pred.size(1)}"
    pred_value, pred_label = pred.topk(maxk, dim=1)
    # transpose to shape (maxk, N, ...)
    pred_label = pred_label.transpose(0, 1)
    correct = pred_label.eq(target.unsqueeze(0).expand_as(pred_label))
    if thresh is not None:
        # Only prediction values larger than thresh are counted as correct
        correct = correct & (pred_value > thresh).t()
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        # NOTE: Change from elenum() to number_of_element()
        # to pass the condition 'fixed_shape=False'
        res.append(correct_k.mul_(100.0 / target.number_of_element()))
    return res[0] if return_single else res


def args_adaptor(np_args):
    pred = torch.from_numpy(np_args[0]).cuda()
    target = torch.from_numpy(np_args[1]).cuda()
    pred.requires_grad = False
    target.requires_grad = False
    topk = 1

    return [pred, target, topk]


def executer_creator():
    return Executer(accuracy, args_adaptor)
