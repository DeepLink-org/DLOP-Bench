import torch
from long_tail_bench.core.executer import Executer


def tblr2bboxes(priors,
                tblr,
                normalizer=4.0,
                normalize_by_wh=True,
                max_shape=None):
    """Decode tblr outputs to prediction boxes.

    The process includes 3 steps: 1) De-normalize tblr coordinates by
    multiplying it with `normalizer`; 2) De-normalize tblr coordinates by the
    prior bbox width and height if `normalize_by_wh` is `True`; 3) Convert
    tblr (top, bottom, left, right) pair relative to the center of priors back
    to (xmin, ymin, xmax, ymax) coordinate.

    Args:
        priors (Tensor): Prior boxes in point form (x0, y0, x1, y1)
          Shape: (n,4).
        tblr (Tensor): Coords of network output in tblr form
          Shape: (n, 4).
        normalizer (Sequence[float] | float): Normalization parameter of
          encoded boxes. By list, it represents the normalization factors at
          tblr dims. By float, it is the unified normalization factor at all
          dims. Default: 4.0
        normalize_by_wh (bool): Whether the tblr coordinates have been
          normalized by the side length (wh) of prior bboxes.
        max_shape (tuple, optional): Shape of the image. Decoded bboxes
          exceeding which will be clamped.

    Return:
        encoded boxes (Tensor), Shape: (n, 4)
    """
    if not isinstance(normalizer, float):
        normalizer = torch.tensor(normalizer, device=priors.device)
        assert len(normalizer) == 4, "Normalizer must have length = 4"
    assert priors.size(0) == tblr.size(0)
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
        boxes[:, 0].clamp_(min=0, max=max_shape[1])
        boxes[:, 1].clamp_(min=0, max=max_shape[0])
        boxes[:, 2].clamp_(min=0, max=max_shape[1])
        boxes[:, 3].clamp_(min=0, max=max_shape[0])
    return boxes


def args_adaptor(np_args):
    priors = torch.from_numpy(np_args[0]).cuda()
    tblr = torch.from_numpy(np_args[1]).cuda()

    return [priors, tblr]


def executer_creator():
    return Executer(tblr2bboxes, args_adaptor)
