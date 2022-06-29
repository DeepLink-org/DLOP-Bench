import os
import sys
from typing import FrozenSet
from random import randint
from .settings import FRAMEWORK, SAMPLE_IMPL, FrameType
from long_tail_bench.core.executer import tensor_type, trans_tensor_to_np
from long_tail_bench.core import CaseFetcher, registry


def random_shape(dims, max_num_per_dim=10):
    assert type(dims) == int
    assert dims > 0
    assert max_num_per_dim >= 1
    shape = [randint(1, max_num_per_dim) for _ in range(dims)]
    return tuple(shape)


def import_impl(sample_module_name, impl):
    import_str = sample_module_name + "." + impl
    try:
        __import__(import_str)
    except ModuleNotFoundError:
        # print(import_str, " is not implemented.")
        print(end="")
    else:
        return sys.modules[import_str]


def trans_to_np(val):
    if isinstance(val, (list, tuple)):
        return tuple(trans_to_np(v) for v in val)
    elif isinstance(val, (set, frozenset)):
        return frozenset(trans_to_np(v) for v in val)
    elif isinstance(val, (dict, FrozenSet)):
        return FrozenSet((k, trans_to_np(v)) for k, v in val.items())
    elif isinstance(val, tensor_type()):
        return trans_tensor_to_np(val)
    else:
        return val


def auto_import(sample_module_name):
    impl = FRAMEWORK if SAMPLE_IMPL is None else SAMPLE_IMPL
    if impl is FrameType.Parrots:
        return import_impl(sample_module_name, "pat_impl")
    elif impl is FrameType.Torch:
        return import_impl(sample_module_name, "torch_impl")
    elif impl is FrameType.XLA:
        return import_impl(sample_module_name, "xla_impl")
    elif impl is FrameType.JAX:
        return import_impl(sample_module_name, "jax_impl")
    else:
        raise Exception("Not set environment variable FRAMEWORK.")


def auto_register(sample_dirpath, sample_module_name):
    for _, dirs, _ in os.walk(sample_dirpath):
        for dir in dirs:
            if dir == "__pycache__":
                continue
            import_impl(sample_module_name, dir)


def register_sample(sample_module_name, get_sample_config, gen_np_args=None):
    backend = auto_import(sample_module_name)
    sample_name = str(sample_module_name).split(".")[-1]
    if backend is not None:
        registry.register(
            sample_name,
            CaseFetcher(backend.executer_creator, get_sample_config,
                        gen_np_args),
        )
