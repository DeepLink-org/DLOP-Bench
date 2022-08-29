from long_tail_bench.core.executer import tensor_type
from long_tail_bench.common import build_pytree


class AbstractTensor(object):

    def __init__(self, tensor):
        self.shape = list(tensor.shape)
        self.dtype = tensor.dtype
        self.requires_grad = tensor.requires_grad


def abstract_tree(args):
    abstract_tensors = []

    def abstract_leaf(leaf):
        if isinstance(leaf, tensor_type()):
            at = AbstractTensor(leaf)
            abstract_tensors.append(at)
            return at
        else:
            return leaf

    return build_pytree(args, abstract_leaf), abstract_tensors
