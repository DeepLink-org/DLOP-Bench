from jax import numpy as np
import jax.numpy as jnp
from jax import device_put
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
        normalizer = jnp.tensor(normalizer, device=priors.device)
        assert len(normalizer) == 4, 'Normalizer must have length = 4'
    #assert priors.size(0) == tblr.size(0)
    loc_decode = tblr * normalizer
    prior_centers = (priors[:, 0:2] + priors[:, 2:4]) / 2
    if normalize_by_wh:
        wh = priors[:, 2:4] - priors[:, 0:2]
        w, h = jnp.split(wh, 2, axis=1)
        '''loc_decode[:, :2] *= h  # tb
        loc_decode[:, 2:] *= w  # lr'''
        loc_decode.at[:, :2].mul(h)
        loc_decode.at[:, 2:].mul(w)
    top, bottom, left, right = loc_decode.split(4, axis=1)
    xmin = jnp.expand_dims(prior_centers[:, 0], 1) - left
    xmax = jnp.expand_dims(prior_centers[:, 0], 1) + right
    ymin = jnp.expand_dims(prior_centers[:, 1], 1) - top
    ymax = jnp.expand_dims(prior_centers[:, 1], 1) + bottom
    boxes = jnp.stack((xmin, ymin, xmax, ymax), axis=1)
    if max_shape is not None:
        boxes[:, 0].clamp(min=0, max=max_shape[1])
        boxes[:, 1].clamp(min=0, max=max_shape[0])
        boxes[:, 2].clamp(min=0, max=max_shape[1])
        boxes[:, 3].clamp(min=0, max=max_shape[0])
    return boxes


def args_adaptor(np_args):
    priors = device_put(np_args[0])
    tblr = device_put(np_args[1])

    return [priors, tblr]


def executer_creator():
    return Executer(tblr2bboxes, args_adaptor)
