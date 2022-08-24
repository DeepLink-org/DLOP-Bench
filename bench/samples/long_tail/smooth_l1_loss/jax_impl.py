import jax
import jax.numpy as jnp
from jax import device_put, grad
from bench.core.executer import Executer


def smooth_l1_loss(pred, target, beta=1.0):
    """Smooth L1 loss.

    Args:
        pred (Tensor): The prediction.
        target (Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        Tensor: Calculated loss
    """
    diff = jax.lax.abs(pred - target)
    loss = jnp.where(diff < beta, 0.5 * diff * diff / beta,
                     diff - 0.5 * beta)
    return jnp.sum(loss)


def args_adaptor(np_args):
    pred = device_put(np_args[0])
    target = device_put(np_args[1])

    return [pred, target]


def executer_creator():
    return Executer(smooth_l1_loss, args_adaptor)
