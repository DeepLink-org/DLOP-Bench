from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json


def get_sample_config():
    with open("./bench/samples/basic/lt/lt.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["lt_0"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["lt_0"][i], arg_data["lt_1"][i]))
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 2,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="",  # noqa
        tags=[SampleTag.ViewAttribute],
    )


def gen_np_args(lt_0, lt_1):
    lt_0 = np.random.random(lt_0)
    if isinstance(lt_1, list): # lt_1 is a tensor
        lt_1 = np.random.random(lt_1)
    else: # lt_1 is an imm
        lt_1 = np.array(lt_1)

    return [lt_0, lt_1]


register_sample(__name__, get_sample_config, gen_np_args)
