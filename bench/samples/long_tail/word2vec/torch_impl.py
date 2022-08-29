import torch
import torch.nn as nn
from long_tail_bench.core.executer import Executer


class Word2VecLayer(nn.Module):
    def __init__(self, sparse_feature_number, emb_dim, neg_num, emb_name,
                 emb_w_name, emb_b_name):
        super(Word2VecLayer, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.emb_dim = emb_dim
        self.neg_num = neg_num
        self.emb_name = emb_name
        self.emb_w_name = emb_w_name
        self.emb_b_name = emb_b_name

        # init_width = 0.5 / self.emb_dim
        self.embedding = torch.nn.Embedding(
            num_embeddings=self.sparse_feature_number,
            embedding_dim=self.emb_dim,
            sparse=True)

        self.embedding_w = torch.nn.Embedding(self.sparse_feature_number,
                                              self.emb_dim,
                                              sparse=True)

        self.embedding_b = torch.nn.Embedding(self.sparse_feature_number,
                                              1,
                                              sparse=True)

    def forward(self, inputs):
        input_emb = self.embedding(inputs[0])
        true_emb_w = self.embedding_w(inputs[1])
        true_emb_b = self.embedding_b(inputs[1])
        input_emb = torch.squeeze(input_emb, 1)
        true_emb_w = torch.squeeze(true_emb_w, 1)
        true_emb_b = torch.squeeze(true_emb_b, 1)

        neg_emb_w = self.embedding_w(inputs[2])
        neg_emb_b = self.embedding_b(inputs[2])

        neg_emb_b_vec = torch.reshape(neg_emb_b, [-1, self.neg_num])

        true_logits = torch.add(
            torch.sum(x=torch.multiply(input_emb, true_emb_w),
                      dim=1,
                      keepdim=True), true_emb_b)

        input_emb_re = torch.reshape(input_emb, [-1, 1, self.emb_dim])
        neg_matmul = torch.matmul(input_emb_re, neg_emb_w).transpose(1, 2)
        neg_matmul_re = torch.reshape(neg_matmul, [-1, self.neg_num])
        neg_logits = torch.add(neg_matmul_re, neg_emb_b_vec)

        return true_logits, neg_logits


def args_adaptor(np_args):
    input = []
    tensor1 = torch.from_numpy(np_args[0]).long()
    tensor2 = torch.from_numpy(np_args[1]).long()
    tensor3 = torch.from_numpy(np_args[2]).long()
    input.append(tensor1)
    input.append(tensor2)
    input.append(tensor3)
    return [input]


def executer_creator():
    n, d, m = 12, 1, 12
    coder_instance = Word2VecLayer(n, d, m, "a", "b", "c")
    return Executer(coder_instance.forward, args_adaptor)
