import torch
import torch.nn
import numpy as np
from long_tail_bench.core.executer import Executer


def mhaf(query, key, value, embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias, bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight, out_proj_bias):

    return torch.nn.functional.multi_head_attention_forward(query, key, value, embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias, bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight, out_proj_bias)


def args_adaptor(np_args):
    query_torch = torch.from_numpy(np_args[0]).to(torch.float32).cuda()
    key_torch = torch.from_numpy(np_args[1]).to(torch.float32).cuda()
    value_torch = torch.from_numpy(np_args[2]).to(torch.float32).cuda()
    embed_dim_to_check = np_args[3]
    num_heads = np_args[4]
    in_proj_weight_np = torch.from_numpy(np_args[5]).to(torch.float32).cuda()
    in_proj_bias_np = torch.from_numpy(np_args[6]).to(torch.float32).cuda()
    bias_k = np_args[7]
    bias_v = np_args[8]
    add_zero_attn_np = np_args[9]
    dropout_p = np_args[10]
    out_proj_weight_np = torch.from_numpy(np_args[11]).to(torch.float32).cuda()
    out_proj_bias_np = torch.from_numpy(np_args[12]).to(torch.float32).cuda()
    return [query_torch, key_torch, value_torch, embed_dim_to_check, num_heads, in_proj_weight_np, in_proj_bias_np, bias_k, bias_v, add_zero_attn_np, dropout_p, out_proj_weight_np, out_proj_bias_np]


def executer_creator():
    return Executer(mhaf, args_adaptor)
