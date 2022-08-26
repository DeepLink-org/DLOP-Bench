from collections import OrderedDict
from bench.core.registrator import custom_class_manager


def build_pytree(tree, func):
    """Search the element which is not a container in tree and 
    put it into func to execute.

    Args:
        tree(Any): Container or any other element.
        func(Function): A fcuntion to execute. 
    """
    if isinstance(tree, (list, tuple, set, frozenset)):
        return type(tree)(build_pytree(sub, func) for sub in tree)
    elif isinstance(tree, (dict, OrderedDict)):
        return type(tree)(
            (key, build_pytree(sub, func)) for key, sub in tree.items())
    else:
        return func(tree)


def unfold_custom_class(node):
    """Identify user custom class registered and transform it to dict.

    Args:
        node(Any): Container or any other element.
    Returns:
        Any or dict: the dict is a container transformed from custom class
    """
    if isinstance(node, tuple(custom_class_manager.get_custom_classes())):
        return dict((key, build_pytree(value, unfold_custom_class))
                    for key, value in node.__dict__.items() if "__" not in key)
    else:
        return node
