import numpy as np
import jax.numpy as jnp
from jax import device_put
from long_tail_bench.core.executer import Executer


def map_roi_levels(rois, num_levels=4, finest_scale=56):
    """Map rois to corresponding feature levels by scales.

    - scale < finest_scale * 2: level 0
    - finest_scale * 2 <= scale < finest_scale * 4: level 1
    - finest_scale * 4 <= scale < finest_scale * 8: level 2
    - scale >= finest_scale * 8: level 3

    Args:
        rois (Tensor): Input RoIs, shape (k, 5).
        num_levels (int): Total level number.

    Returns:
        Tensor: Level index (0-based) of each RoI, shape (k, )
    """
    scale = jnp.sqrt(
        (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
    target_lvls = jnp.floor(jnp.log2(scale / finest_scale + 1e-6))
    # target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()     # jax没有clamp接口
    target_lvls = target_lvls.clip(a_min=0, a_max=num_levels - 1).astype(jnp.int32)

    return target_lvls


def args_adaptor(np_args):
    rois = device_put(np_args[0])
    return [rois, np_args[1], np_args[2]]


def executer_creator():
    return Executer(map_roi_levels, args_adaptor)
