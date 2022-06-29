import jax.numpy as jnp
from jax import jit, device_put
from long_tail_bench.core.executer import Executer


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = jnp.split(x, 4, axis=-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return jnp.stack(b, axis=-1)


def args_adaptor(np_args):
    x = device_put(np_args[0])
    return [x]


def executer_creator():
    return Executer(box_xyxy_to_cxcywh, args_adaptor)
