import torch
import string

float_types = [
    torch.float32, torch.float, torch.float64, torch.double, torch.float16,
    torch.half
]
int_types = [
    torch.uint8, torch.int8, torch.int16, torch.short, torch.int, torch.int32,
    torch.int64, torch.long, torch.bool
]


class NameGenerator(object):
    _count = 0
    _instance = None
    _list_alphabet = list(string.ascii_lowercase)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    @classmethod
    def get_a_name(cls):
        assert cls._count < len(cls._list_alphabet)
        name = cls._list_alphabet[cls._count]
        cls._count = cls._count + 1
        return name


name_generator = NameGenerator()


def get_init_import():
    return """from long_tail_bench.common import (
                SampleConfig,
                register_sample,
                SampleSource,
                SampleTag,
            )
import numpy as np"""


def get_impl_import():
    return """import torch
from long_tail_bench.core.executer import Executer"""


def get_sample_config(out_len, in_len):
    return """def get_sample_config():
                  return SampleConfig(
                    args_cases=[],
                    requires_grad=[False] * {},
                    backward=[False] * {},
                    performance_iters=1000,
                    save_timeline=False,
                    source=SampleSource.UNKNOWN,
                    url="",  # noqa
                    tags=[],
                )""".format(in_len, out_len)


def gen_base_np(shape, dtype):
    name = name_generator.get_a_name()
    s_shape = [str(d) for d in shape]
    if dtype in float_types:
        return """{} = np.random.randn({})
    {} = {}.astype(np.float32)""".format(name, ",".join(s_shape), name,
                                         name), name
    elif dtype in int_types:
        return """{} = np.random.randint(10, size=({}))
    {} = {}.astype(np.float32)""".format(name, ",".join(s_shape), name,
                                         name), name
    else:
        raise Exception("Not support dtype:{}".format(dtype))


def gen_np_args(inputs):
    names = []
    core_codes = []
    for at in inputs:
        one_np, name = gen_base_np(at.shape, at.dtype)
        names.append(name)
        core_codes.append(one_np)
    return """def gen_np_args():
    {}
    return [{}]""".format("\n".join(core_codes), ",".join(names)), names


def get_register_sample():
    return """register_sample(__name__, get_sample_config, gen_np_args)"""


def get_args_adaptor(inputs, names):
    core_codes = []
    for idx, at in enumerate(inputs):
        core_codes.append("{} = torch.from_numpy(np_args[{}]).cuda()".format(
            names[idx], idx))

    return """def args_adaptor(np_args):
    {}
    return [{}]""".format("\n".join(core_codes), ",".join(names))


def gen_executer_creator(entry_name):
    return """def executer_creator():
    return Executer({}, args_adaptor)""".format(entry_name)
