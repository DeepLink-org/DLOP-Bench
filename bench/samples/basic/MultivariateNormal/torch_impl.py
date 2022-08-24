import torch
import numpy as np
from bench.core.executer import Executer


def MultivariateNormal(loc, covariance_matrix):
    sampler = torch.distributions.MultivariateNormal(loc=loc, covariance_matrix=covariance_matrix)
    ret = sampler.sample()
    return ret

def args_adaptor(np_args):
    np_loc = np_args[0]
    np_cov_mat = np_args[1]
    loc = torch.from_numpy(np_loc).to("cuda")
    covariance_matrix = torch.from_numpy(np_cov_mat).to("cuda")
    return [loc, covariance_matrix]

def executer_creator():
    return Executer(MultivariateNormal, args_adaptor)