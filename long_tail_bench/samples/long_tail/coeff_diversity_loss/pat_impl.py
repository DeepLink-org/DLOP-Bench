import torch
import torch.nn.functional as F
from long_tail_bench.core.executer import Executer


def coeff_diversity_loss(coeffs, instance_t):
    """
        coeffs     should be size [num_pos, num_coeffs]
        instance_t should be size [num_pos]
                   and be values from 0 to num_instances-1
        """
    num_pos = coeffs.size(0)
    instance_t = instance_t.view(-1)  # juuuust to make sure

    coeffs_norm = F.normalize(coeffs, dim=1)
    cos_sim = coeffs_norm @ coeffs_norm.t()

    inst_eq = (instance_t[:, None].expand_as(cos_sim) == instance_t[
        None, :].expand_as(cos_sim)).float()

    # Rescale to be between 0 and 1
    cos_sim = (cos_sim + 1) / 2

    # If they're the same instance, use cosine distance, else use cosine
    # similarity
    loss = (1 - cos_sim) * inst_eq + cos_sim * (1 - inst_eq)

    # Only divide by num_pos once
    # because we're summing over a num_pos x num_pos tensor
    # and all the losses will be divided by num_pos at the end,
    # so just one extra time.
    # return cfg.mask_proto_coeff_diversity_alpha * loss.sum() / num_pos
    return 1 * loss.sum() / num_pos


def args_adaptor(np_args):
    coeffs = torch.from_numpy(np_args[0]).cuda()
    instance_t = torch.from_numpy(np_args[1]).cuda()
    return [coeffs, instance_t]


def executer_creator():
    return Executer(coeff_diversity_loss, args_adaptor)
