import jax.numpy as jnp
from jax import device_put
from bench.core.executer import Executer


def accuracy(output, target, topk=(1,), raw=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)

    # _, pred = output.topk(maxk, 1, True, True)
    pred = jnp.argsort(output, axis=1)[:, :-maxk-1:-1]
    pred = jnp.transpose(pred)
    correct = jnp.equal(jnp.reshape(target, (1, -1)), pred)

    res = []
    for k in topk:
        correct_k = jnp.sum(jnp.reshape(correct[:k], [-1]).astype('float32'), axis=0, keepdims=True)
        if raw:
            res.append(correct_k)
        else:
            res.append(jnp.multiply(correct_k, 100.0 / target.shape[0]))
    return res

def args_adaptor(np_args):
    output = device_put(np_args[0])
    target = device_put(np_args[1])
    return [output, target, (1, 5), False]


def executer_creator():
    return Executer(accuracy, args_adaptor)
