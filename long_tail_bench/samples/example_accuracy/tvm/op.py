import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

import torch.nn as nn
import torch.nn.init as init
from gen_data import gen_np_args, args_adaptor

import onnx
# import onnxoptimizer

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


class Bbox(nn.Module):
    def __init__(self):
        super(Bbox, self).__init__()
        self.accuracy = accuracy

    def forward(self, pred, target):
        loss = self.accuracy(pred, target)

        return loss 

torch_model = Bbox()

torch_model.eval()

output, target = args_adaptor(gen_np_args(32, 1000))
torch_out = torch_model(output, target)

torch.onnx.export(torch_model, 
        (output, target),
        "accuracy.onnx",
        verbose=True,
        export_params=True,
        opset_version=12,
        input_names=['output', 'target'],
        output_names = ['output'])
