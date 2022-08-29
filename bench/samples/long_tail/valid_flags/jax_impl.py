import jax.numpy as jnp
from jax import jit, device_put
from long_tail_bench.core.executer import Executer


def meshgrid(x, y, row_major=True):
    xx = x.repeat(len(y))
    yy = y.reshape(-1, 1).tile(len(x)).reshape(-1) #np.tile <- jnp.repeat
    if row_major:
        return xx, yy
    else:
        return yy, xx


def valid_flags(featmap_size, valid_size, num_base_anchors):
    feat_h, feat_w = featmap_size
    valid_h, valid_w = valid_size
    assert valid_h <= feat_h and valid_w <= feat_w
    valid_x = jnp.zeros(feat_w, dtype=jnp.uint8)
    valid_y = jnp.zeros(feat_h, dtype=jnp.uint8)
    valid_x.at[:valid_w].set(1)
    valid_y.at[:valid_h].set(1)
    valid_xx, valid_yy = meshgrid(valid_x, valid_y)
    valid = valid_xx & valid_yy
    valid = valid[:, None]
    num_base_anchors = 2
    jnp.expand_dims(valid, num_base_anchors).reshape(-1)
    return valid


def args_adaptor(np_args):
    featmap_size = device_put(np_args[0])
    valid_size = device_put(np_args[1])
    num_base_anchors = np_args[2]
    return [featmap_size, valid_size, num_base_anchors]


def executer_creator():
    return Executer(valid_flags, args_adaptor)
