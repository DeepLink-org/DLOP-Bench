from bench.common import SampleConfig, SampleSource, SampleTag
import numpy as np


def get_sample_config():
    return SampleConfig(
        args_cases=[(10, 8, 4), (8, 6, 4), (12, 8, 4)],
        requires_grad=[True, True, False],
        backward=[False, False],
        performance_iters=1000,
        save_timeline=False,
        source=SampleSource.MMDET,
        url="https://github.com/open-mmlab/mmdetection/blob/c76ab0eb3c637b86c343d8454e07e00cfecc1b78/mmdet/models/losses/ae_loss.py#L11",  # noqa
        tags=[
            SampleTag.Reduce,
            SampleTag.IfElseBranch,
            SampleTag.ViewAttribute,
        ],
    )


def gen_np_args(M, N, K):
    tl_preds = np.random.randn(M, M, M, 2, 1)
    tl_preds = tl_preds.astype(np.float32)

    br_preds = np.random.randn(M, M, M, 2, 1)
    br_preds = br_preds.astype(np.float32)

    match = []
    match_point_0 = [[N, N], [K, K]]
    match.append(match_point_0)
    match_point_1 = [[int(N / 2), int(N / 2)], [int(K / 2), int(K / 2)]]
    match.append(match_point_1)
    match_point_2 = [[int(N / 3), int(N / 3)], [int(K / 3), int(K / 3)]]
    match.append(match_point_2)
    match_point_3 = [[int(N / 4), int(N / 4)], [int(K / 4), int(K / 4)]]
    match.append(match_point_3)
    return [tl_preds, br_preds, match]


# This case stuck in S3, so we skip it temporarily
# register_sample(__name__, get_sample_config, gen_np_args)
