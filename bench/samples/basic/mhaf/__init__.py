# Copyright (c) OpenComputeLab. All Rights Reserved.

from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np
import json


def get_sample_config():
    with open("./bench/samples/basic/mhaf/mhaf.json", "r") as f:
        arg_data = json.load(f)
    arg_data_length = len(arg_data["query"])
    args_cases_ = []
    for i in range(arg_data_length):
        args_cases_.append((arg_data["query"][i], arg_data["key"][i], arg_data["value"][i], arg_data["embed_dim_to_check"][i], arg_data["num_heads"][i], arg_data["in_proj_weight"][i], arg_data["in_proj_bias"][i], arg_data["bias_k"][i], arg_data["bias_v"][i], arg_data["add_zero_attn"][i], arg_data["dropout_p"][i], arg_data["out_proj_weight"][i], arg_data["out_proj_bias"][i]))
    return SampleConfig(
        args_cases=args_cases_,
        requires_grad=[False] * 13,
        backward=False,
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="",  # noqa
        tags=[SampleTag.ViewAttribute],
    )


def gen_np_args(query_size, key_size, value_size, embed_dim_to_check_, num_heads_, in_proj_weight_size, in_proj_bias_size, bias_k_, bias_v_, add_zero_attn_, dropout_p_, out_proj_weight_size, out_proj_bias_size):
    query_np = np.random.random(query_size)
    key_np = np.random.random(key_size)
    value_np = np.random.random(value_size)
    embed_dim_to_check = embed_dim_to_check_[0]
    num_heads = num_heads_[0]
    in_proj_weight_np = np.random.random(in_proj_weight_size)
    in_proj_bias_np = np.random.random(in_proj_bias_size)
    bias_k = bias_k_[0]
    bias_v = bias_v_[0]
    add_zero_attn_np = add_zero_attn_[0]
    dropout_p = dropout_p_[0]
    out_proj_weight_np = np.random.random(out_proj_weight_size)
    out_proj_bias_np = np.random.random(out_proj_bias_size)

    return [query_np, key_np, value_np, embed_dim_to_check, num_heads, in_proj_weight_np, in_proj_bias_np, bias_k, bias_v, add_zero_attn_np, dropout_p, out_proj_weight_np, out_proj_bias_np]


register_sample(__name__, get_sample_config, gen_np_args)
