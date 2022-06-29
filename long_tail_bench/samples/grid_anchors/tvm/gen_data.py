import numpy as np
import torch


def gen_np_args(M, N, K):
    def gen_base(row):
        data = np.random.randn(row, 4)
        data = data.astype(np.float32)
        return data

    base_anchors = gen_base(M)

    return [base_anchors]

def args_adaptor(np_args):
    base_anchors = torch.from_numpy(np_args[0]).cuda()

    return [base_anchors]
