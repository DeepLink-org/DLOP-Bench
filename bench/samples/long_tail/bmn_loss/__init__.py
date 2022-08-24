from bench.common import (
    SampleConfig,
    register_sample,
    SampleSource,
    SampleTag,
)
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(8, 32), (16, 16), (16, 8)],
        requires_grad=[True, True, True, False, False, False, False],
        backward=[True, False, False, False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMACTION2,
        url="https://github.com/open-mmlab/mmaction2/blob/d5ab34805fe6a02bb98e4af158626a21790b6974/mmaction/models/losses/bmn_loss.py#L11",  # noqa
        tags=[SampleTag.ViewAttribute, SampleTag.Reduce],
    )


def gen_np_args(M, N):
    def gen_base(shape):
        data = np.random.randn(*shape)
        data = data.astype(np.float32)
        return data

    shape1 = (M, 2, N, N)
    shape2 = (M, N, N)
    shape3 = (M, N)
    shape4 = (N, N)
    pred_bm = gen_base(shape1)
    pred_start = gen_base(shape3)
    pred_end = gen_base(shape3)
    gt_iou_map = gen_base(shape2)
    gt_start = gen_base(shape3)
    gt_end = gen_base(shape3)
    bm_mask = gen_base(shape4)

    return [
        pred_bm,
        pred_start,
        pred_end,
        gt_iou_map,
        gt_start,
        gt_end,
        bm_mask,
    ]


register_sample(__name__, get_sample_config, gen_np_args)
