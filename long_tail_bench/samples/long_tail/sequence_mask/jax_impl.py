import jax.numpy as jnp
import jax
from jax import device_put
from long_tail_bench.core.executer import Executer


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.size
    max_len = max_len or lengths.max()
    max_len_tensor = jnp.arange(0, max_len, dtype=lengths.dtype)
    max_len_tensor = jnp.repeat(max_len_tensor, batch_size, 0)
    max_len_tensor = jnp.expand_dims(max_len_tensor, 1)
    lengths = jnp.repeat(lengths, int(max_len), 0)
    lengths =  jnp.expand_dims(lengths, 1)
    return jax.lax.lt(max_len_tensor, lengths)


def args_adaptor(np_args):
    input0 = device_put(np_args[0])
    return [input0]


def executer_creator():
    return Executer(sequence_mask, args_adaptor)
