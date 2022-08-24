import os
import inspect
from abc import abstractmethod
from .abstract_tree import abstract_tree
from .autogen import (get_sample_config, gen_np_args, get_args_adaptor,
                      gen_executer_creator, get_init_import,
                      get_register_sample, get_impl_import)


class BaseGripper(object):

    def __init__(self, dist_dir):
        self._dist_dir = os.path.dirname(__file__).replace(
            "tools", "samples") if dist_dir is None else dist_dir
        self._signature = None
        self._grapped = False
        self._inputs = None
        self._inputs_tree = None
        self._outputs = None
        self._outputs_tree = None
        self._get_sample_config = None
        self._gen_np_args = None
        self._body = None
        self._args_adaptor = None
        self._executer_creator = None
        self._init = None
        self._pat_impl = None

    @abstractmethod
    def entry_name(self):
        pass

    @property
    def grapped(self):
        return self._grapped

    @grapped.setter
    def grapped(self, value):
        self._grapped = value

    def analyse_inputs(self, *args, **kwargs):
        if kwargs:
            bound_args = self._signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            args = bound_args.args
        self._inputs_tree, self._inputs = abstract_tree(args)

    def analyse_outputs(self, outputs):
        self._outputs_tree, self._outputs = abstract_tree(outputs)

    @abstractmethod
    def analyse_body(self):
        pass

    def format(self):
        """
        Transfer inputs outputs and class or func codes to format we need,
        such as str of function `get_sample_config`,
        `gen_np_args`, `args_adaptor`,
        `executer_creator` and dist class.
        """
        self._get_sample_config = get_sample_config(len(self._outputs),
                                                    len(self._inputs))
        self._gen_np_args, names = gen_np_args(self._inputs)
        self._args_adaptor = get_args_adaptor(self._inputs, names)
        self._executer_creator = gen_executer_creator(self.entry_name())
        self._init = get_init_import(
        ) + "\n\n" + self._get_sample_config + "\n\n" + self._gen_np_args + "\n\n" + get_register_sample(  # noqa
        )
        self._pat_impl = get_impl_import(
        ) + "\n\n" + self._body + "\n\n" + self._args_adaptor + "\n\n" + self._executer_creator  # noqa

    def save_to_dist_dir(self):
        sample_path = os.path.join(self._dist_dir, self.entry_name())
        assert not os.path.exists(sample_path)
        os.mkdir(sample_path)
        with open(os.path.join(sample_path, '__init__.py'),
                  'w',
                  encoding='utf-8') as f:
            f.write(self._init)
        with open(os.path.join(sample_path, 'pat_impl.py'),
                  'w',
                  encoding='utf-8') as f:
            f.write(self._pat_impl)


class ClassGripper(BaseGripper):

    def __init__(self, cls, dist_dir, class_entry):
        super().__init__(dist_dir)
        self._cls = cls
        self._cls_entry = class_entry
        self._signature = inspect.signature(getattr(self._cls,
                                                    self._cls_entry),
                                            follow_wrapped=False)

    def entry_name(self):
        return self._cls_entry

    @property
    def class_entry(self):
        return self._cls_entry

    @class_entry.setter
    def class_entry(self, value):
        self._cls_entry = value

    def analyse_inputs(self, *args, **kwargs):
        """
        Analyse inputs codes.
        """
        super().analyse_inputs(*args[1:], **kwargs)

    def analyse_body(self):
        """
        Analyse class body codes.
        """
        assert self._cls is not None
        self._body = inspect.getsource(self._cls)


class FuncGripper(BaseGripper):

    def __init__(self, func, dist_dir):
        super().__init__(dist_dir)
        self._func = func
        self._signature = inspect.signature(self._func, follow_wrapped=False)

    def entry_name(self):
        return self._func.__name__

    @property
    def func(self):
        return self._func

    @func.setter
    def func(self, value):
        self._func = value

    def analyse_body(self):
        """
        Analyse func body codes.
        """
        assert self._func is not None
        self._body = inspect.getsource(self._func)
        self._body = '\n'.join(self._body.split('\n')[1:])


def wrap_function(func, gripper):

    def wrapped(*args, **kwargs):
        if gripper.grapped:
            return func(*args, **kwargs)
        else:
            gripper.analyse_inputs(*args, **kwargs)
            result = func(*args, **kwargs)
            gripper.analyse_outputs(result)
            gripper.analyse_body()
            gripper.format()
            gripper.save_to_dist_dir()
            gripper.gripped = True
            return result

    return wrapped


def grip(obj=None, dist_dir=None, class_entry=None):
    """
    Grip class or func codes to dist dir.

    Example usage:
    >>> @grip(class_entry="forward")
    ... class Test(obj)
    ...     def __init__(self):
    ...         ...
    ...
    ...
    ...     def forward(self):
    ...         ...

    >>> @grip
    ... def bbox2delta(a, b, c):
    ...     ...
    ...


    Args:
        obj: the class or function you want to grip.
        dist_dir: the path we want to save to.
        class_entry: the entry function name of the class decorated.
    """

    def decorate(obj):
        if inspect.isclass(obj):
            assert class_entry is not None
            assert isinstance(class_entry, str)
            gripper = ClassGripper(obj, dist_dir, class_entry)
            func = getattr(obj, gripper.class_entry)
        elif inspect.isfunction(obj):
            gripper = FuncGripper(obj, dist_dir)
            func = gripper.func
        else:
            raise Exception("Not support  {}".format(type(obj)))

        wrapped_func = wrap_function(func, gripper)
        if inspect.isclass(obj):
            setattr(obj, class_entry, wrapped_func)
            return obj
        elif inspect.isfunction(obj):
            return wrapped_func

    if obj is None:
        return decorate
    else:
        return decorate(obj, dist_dir, class_entry)
