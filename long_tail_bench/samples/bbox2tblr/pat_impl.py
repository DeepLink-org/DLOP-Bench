import torch
from long_tail_bench.core.executer import Executer


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


def args_adaptor(np_args):
    priors = torch.from_numpy(np_args[0]).cuda()
    gts = torch.from_numpy(np_args[1]).cuda()
    return [priors, gts]


def executer_creator():
    return Executer(bboxes2tblr, args_adaptor)
