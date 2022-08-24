import jax.numpy as jnp
from jax import device_put
from bench.core.executer import Executer


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def args_adaptor(np_args):
    boxes = device_put(np_args[0])
    return [boxes]


def executer_creator():
    return Executer(box_area, args_adaptor)
