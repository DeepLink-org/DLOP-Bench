import jax.numpy as jnp
from jax import device_put, lax
from long_tail_bench.core.executer import Executer


def accuracy(pred, target, topk=1, thresh=None):
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    if pred.shape[0] == 0:
        accu = [jnp.zeros(pred.shape) for i in range(len(topk))]
        return accu[0] if return_single else accu
    # pred_value, pred_label = jnp.TopK(pred, maxk) 
    pred_label = jnp.argsort(pred, axis=1)[:, :-maxk-1:-1]
    pred_value = pred[pred_label]
    pred_label = jnp.transpose(pred_label)  # transpose to shape (maxk, N)
    correct = jnp.equal(jnp.reshape(target, (1, -1)), pred_label)
    if thresh is not None:
        # Only prediction values larger than thresh are counted as correct
        correct = lax.bitwise_and(jnp.reshape(correct, (1,) + correct.shape), jnp.transpose(jnp.greater(pred_value, thresh)))
    res = []
    for k in topk:
        correct_k = jnp.sum(jnp.reshape(correct[:k], [-1]).astype('float32'), axis=0, keepdims=True)
        res.append(jnp.multiply(correct_k, 100.0 / pred.shape[0]))
    return res[0] if return_single else res


def args_adaptor(np_args):
    output = device_put(np_args[0])
    target = device_put(np_args[1])

    return [output, target, 1]


def executer_creator():
    return Executer(accuracy, args_adaptor)
