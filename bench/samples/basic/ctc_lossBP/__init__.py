from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json
import random


def get_sample_config():
    with open("./bench/samples/basic/ctc_loss/ctc_loss.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["x1"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["x1"][i], arg_data["x2"][i], 
            arg_data["x3"][i], arg_data["x4"][i], arg_data["x5"][i], 
            arg_data["x6"][i], arg_data["x7"][i]))
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 7,
        backward=[False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.UNKNOWN,
        url="",  # noqa
        tags=[],
    )


def gen_np_args(log_probs_, targets_, input_lengths_,
            target_lengths_, blank_, reduction_, zero_infinity_):
    log_probs = np.random.random(log_probs_).astype(np.float32)
    targets = np.random.randint(1, 20, size=targets_)
    input_lengths = np.full(input_lengths_, log_probs_[0], dtype=np.long)
    val = int(targets_[0]/target_lengths_[0])
    left_val = targets_[0] - target_lengths_[0]*val
    assert len(targets_) == 1, "Should rewrite the data generator \
        for other conditions."
    target_lengths = []
    for i in range(target_lengths_[0]):
        target_lengths.append(val)
    target_lengths[target_lengths_[0]-1] += left_val
    target_lengths = np.array(target_lengths)
    # target_lengths = np.random.randint(1, val, size=target_lengths_, dtype=np.long)
    
    return [log_probs, targets, input_lengths,
            target_lengths, blank_, reduction_, zero_infinity_]


register_sample(__name__, get_sample_config, gen_np_args)
