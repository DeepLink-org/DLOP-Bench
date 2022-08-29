# Copyright (c) OpenComputeLab. All Rights Reserved.
# Modified from OpenMMLab.
import torch

import torch.nn as nn
from bench.core.executer import Executer


class FM(nn.Module):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, sparse_num_field):
        super(FM, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.dense_feature_dim = dense_feature_dim
        self.dense_emb_dim = self.sparse_feature_dim
        self.sparse_num_field = sparse_num_field
        self.init_value_ = 0.1

        use_sparse = True
        # if torch.is_compiled_with_npu():
        #     use_sparse = False

        # sparse part coding
        self.embedding_one = torch.nn.Embedding(sparse_feature_number,
                                                1,
                                                sparse=use_sparse)

        self.embedding = torch.nn.Embedding(self.sparse_feature_number,
                                            self.sparse_feature_dim,
                                            sparse=use_sparse)

        # dense part coding
        self.dense_w_one = torch.nn.Parameter(
            torch.ones(self.dense_feature_dim))
        self.dense_w = torch.nn.Parameter(
            torch.ones((1, self.dense_feature_dim, self.dense_emb_dim)))

    def forward(self, sparse_inputs, dense_inputs):
        # -------------------- first order term  --------------------
        sparse_inputs_concat = torch.concat([sparse_inputs, dense_inputs], 1)
        sparse_emb_one = self.embedding_one(sparse_inputs_concat)

        dense_emb_one = torch.multiply(dense_inputs, self.dense_w_one)
        dense_emb_one = torch.unsqueeze(dense_emb_one, 2)

        y_first_order = torch.sum(sparse_emb_one, 1) + torch.sum(
            dense_emb_one, 1)

        # -------------------- second order term  --------------------
        sparse_embeddings = self.embedding(sparse_inputs_concat)
        dense_inputs_re = torch.unsqueeze(dense_inputs, 2)
        dense_embeddings = torch.multiply(dense_inputs_re, self.dense_w)
        feat_embeddings = torch.concat([sparse_embeddings, dense_embeddings],
                                       1)

        # sum_square part
        summed_features_emb = torch.sum(feat_embeddings,
                                        1)  # None * embedding_size
        summed_features_emb_square = torch.square(
            summed_features_emb)  # None * embedding_size

        # square_sum part
        squared_features_emb = torch.square(
            feat_embeddings)  # None * num_field * embedding_size
        squared_sum_features_emb = torch.sum(squared_features_emb,
                                             1)  # None * embedding_size

        y_second_order = 0.5 * torch.sum(
            summed_features_emb_square - squared_sum_features_emb,
            1,
            keepdim=True)  # None * 1

        return y_first_order, y_second_order


def args_adaptor(np_args):
    sparse_inputs = torch.from_numpy(np_args[0]).long()
    dense_inputs = torch.from_numpy(np_args[1]).long()
    return [sparse_inputs, dense_inputs]


def executer_creator():
    n, d, m = 12, 1, 12
    coder_instance = FM(n, d, m, m)
    return Executer(coder_instance.forward, args_adaptor)
