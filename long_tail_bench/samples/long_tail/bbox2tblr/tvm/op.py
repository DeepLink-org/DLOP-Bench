import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

import torch.nn as nn
import torch.nn.init as init

import onnx
from gen_data import gen_np_args, args_adaptor


def bboxes2tblr(priors, gts, normalizer=4.0, normalize_by_wh=True):
    """Encode ground truth boxes to tblr coordinate.

    It first convert the gt coordinate to tblr format,
     (top, bottom, left, right), relative to prior box centers.
     The tblr coordinate may be normalized by the side length of prior bboxes
     if `normalize_by_wh` is specified as True, and it is then normalized by
     the `normalizer` factor.

    Args:
        priors (Tensor): Prior boxes in point form
            Shape: (num_proposals,4).
        gts (Tensor): Coords of ground truth for each prior in point-form
            Shape: (num_proposals, 4).
        normalizer (Sequence[float] | float): normalization parameter of
            encoded boxes. If it is a list, it has to have length = 4.
            Default: 4.0
        normalize_by_wh (bool): Whether to normalize tblr coordinate by the
            side length (wh) of prior bboxes.

    Return:
        encoded boxes (Tensor), Shape: (num_proposals, 4)
    """

    # dist b/t match center and prior's center
    if not isinstance(normalizer, float):
        normalizer = torch.tensor(normalizer, device=priors.device)
        assert len(normalizer) == 4, "Normalizer must have length = 4"
    assert priors.size(0) == gts.size(0)
    prior_centers = (priors[:, 0:2] + priors[:, 2:4]) / 2
    xmin, ymin, xmax, ymax = gts.split(1, dim=1)
    top = prior_centers[:, 1].unsqueeze(1) - ymin
    bottom = ymax - prior_centers[:, 1].unsqueeze(1)
    left = prior_centers[:, 0].unsqueeze(1) - xmin
    right = xmax - prior_centers[:, 0].unsqueeze(1)
    loc = torch.cat((top, bottom, left, right), dim=1)
    if normalize_by_wh:
        # Normalize tblr by anchor width and height
        wh = priors[:, 2:4] - priors[:, 0:2]
        w, h = torch.split(wh, 1, dim=1)
        loc[:, :2] /= h  # tb is normalized by h
        loc[:, 2:] /= w  # lr is normalized by w
    # Normalize tblr by the given normalization factor
    return loc / normalizer


class Bbox(nn.Module):
    def __init__(self):
        super(Bbox, self).__init__()
        self.bboxes2tblr = bboxes2tblr

    def forward(self, priors, gts):
        return self.bboxes2tblr(priors, gts)

torch_model = Bbox()
torch_model.eval()

priors, gts = args_adaptor(gen_np_args(3000, 4))
torch_out = torch_model(priors, gts)

torch.onnx.export(torch_model,
        (priors, gts),
        "bbox2tblr.onnx",
        verbose=True,
        export_params=True,
        opset_version=12,
        input_names=['priors', 'gts'],
        output_names = ['output'])
