import torch
from long_tail_bench.core.executer import Executer


def accuracy(output, target, topk=(1, ), raw=False):
    """
    Computes the accuracy over the k top predictions for the specified values
    of k"""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            if raw:
                res.append(correct_k)
            else:
                res.append(correct_k.mul(100.0 / target.size(0)))
        return res


def args_adaptor(np_args):
    output = torch.from_numpy(np_args[0]).cuda()
    target = torch.from_numpy(np_args[1]).cuda()
    return [output, target, (1, 5)]


def executer_creator():
    return Executer(accuracy, args_adaptor)
