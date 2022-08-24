import jax.numpy as jnp
from jax import jit, device_put
from bench.core.executer import Executer

def gen_base_anchors(base_size, ratios, scales):
    w = base_size
    h = base_size
    x_ctr = 0.5 * (w - 1)
    y_ctr = 0.5 * (h - 1)

    h_ratios = jnp.sqrt(ratios)
    w_ratios = 1 / h_ratios
    ws = (w * w_ratios[:, None] * scales[None, :]).reshape(-1)
    hs = (h * h_ratios[:, None] * scales[None, :]).reshape(-1)

    base_anchors = jnp.stack(
        [
            x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
        ],
        axis=-1).round()

    return base_anchors


def args_adaptor(np_args):
    base_size = np_args[0]
    ratios = device_put(np_args[1])
    scales = device_put(np_args[2])
    return [base_size, ratios, scales]


def executer_creator():
    return Executer(gen_base_anchors, args_adaptor)
