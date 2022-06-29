from collections import OrderedDict
from long_tail_bench.core.registrator import custom_class_manager


def build_pytree(tree, func):
    if isinstance(tree, (list, tuple, set, frozenset)):
        return type(tree)(build_pytree(sub, func) for sub in tree)
    elif isinstance(tree, (dict, OrderedDict)):
        return type(tree)(
            (key, build_pytree(sub, func)) for key, sub in tree.items())
    else:
        return func(tree)


def unfold_custom_class(node):
    if isinstance(node, tuple(custom_class_manager.get_custom_classes())):
        return dict((key, build_pytree(value, unfold_custom_class))
                    for key, value in node.__dict__.items() if "__" not in key)
    else:
        return node
