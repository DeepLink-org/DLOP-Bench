# Copyright (c) OpenComputeLab. All Rights Reserved.
# Modified from OpenMMLab.

import torch
from bench.core.executer import Executer


def tblr2bboxes(priors,
                tblr,
                normalizer=torch.FloatTensor([4.0]).cuda(),
                normalize_by_wh=torch.IntTensor([1]).cuda(),
                max_shape=torch.IntTensor([0]).cuda()):
    '''if not isinstance(normalizer, float):
        normalizer = torch.tensor(normalizer, device=priors.device)
        assert len(normalizer) == 4, 'Normalizer must have length = 4'
    assert priors.size(0) == tblr.size(0)'''
    loc_decode = tblr * normalizer
    prior_centers = (priors[:, 0:2] + priors[:, 2:4]) / 2
    if normalize_by_wh:
        wh = priors[:, 2:4] - priors[:, 0:2]
        w, h = torch.split(wh, 1, dim=1)
        loc_decode[:, :2] *= h  # tb
        loc_decode[:, 2:] *= w  # lr
    top, bottom, left, right = loc_decode.split(1, dim=1)
    xmin = prior_centers[:, 0].unsqueeze(1) - left
    xmax = prior_centers[:, 0].unsqueeze(1) + right
    ymin = prior_centers[:, 1].unsqueeze(1) - top
    ymax = prior_centers[:, 1].unsqueeze(1) + bottom
    boxes = torch.cat((xmin, ymin, xmax, ymax), dim=1)
    if max_shape is not None:
        return boxes
        boxes[:, 0].clamp_(min=0, max=max_shape[1])
        boxes[:, 1].clamp_(min=0, max=max_shape[0])
        boxes[:, 2].clamp_(min=0, max=max_shape[1])
        boxes[:, 3].clamp_(min=0, max=max_shape[0])
    return boxes


def args_adaptor(np_args):
    priors = torch.from_numpy(np_args[0]).cuda()
    tblr = torch.from_numpy(np_args[1]).cdua()

    return [priors, tblr]


def executer_creator():
    return Executer(tblr2bboxes, args_adaptor)
