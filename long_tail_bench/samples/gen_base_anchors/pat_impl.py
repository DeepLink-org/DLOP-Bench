import torch
from long_tail_bench.core.executer import Executer


def gen_base_anchors(base_size, ratios, scales):
    w = base_size
    h = base_size
    x_ctr = 0.5 * (w - 1)
    y_ctr = 0.5 * (h - 1)

    h_ratios = torch.sqrt(ratios)
    w_ratios = 1 / h_ratios
    ws = (w * w_ratios[:, None] * scales[None, :]).view(-1)
    hs = (h * h_ratios[:, None] * scales[None, :]).view(-1)

    base_anchors = torch.stack(
        [
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1),
        ],
        dim=-1,
    ).round()

    return base_anchors


def args_adaptor(np_args):
    base_size = np_args[0]
    ratios = torch.from_numpy(np_args[1]).cuda()
    scales = torch.from_numpy(np_args[2]).cuda()
    return [base_size, ratios, scales]


def executer_creator():
    return Executer(gen_base_anchors, args_adaptor)
