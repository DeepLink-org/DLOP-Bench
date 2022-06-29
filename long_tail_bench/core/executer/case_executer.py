from abc import abstractmethod, ABC


class AbstractExecuter(ABC):
    @abstractmethod
    def generate_args(self, case, requires_grad):
        pass

    @abstractmethod
    def prepare(self, stage_mode):
        pass

    @abstractmethod
    def clone_func_args(self, func_args):
        pass

    @abstractmethod
    def execute(self, func_args):
        pass

    @abstractmethod
    def synchronize(self):
        pass

    @abstractmethod
    def assert_correct(self, val_a, val_b):
        pass

    @abstractmethod
    def backward(self, ret_t, func_arg):
        pass

    @abstractmethod
    def reset_grad(self, func_args, requires_grad):
        pass

    @abstractmethod
    def save_timeline_start(self):
        pass

    @abstractmethod
    def save_timeline_end(self):
        pass

    @abstractmethod
    def register_custom_class(self, obj):
        pass


class BaseCaseExecuter(AbstractExecuter):
    def __init__(self, core_func, args_adaptor):
        self._origin_func = core_func
        self._args_adaptor = args_adaptor
        self._execute_func = None

    def adapt_args(self, np_args):
        assert isinstance(np_args, list)
        return self._args_adaptor(np_args)

    def execute(self, func_args):
        assert isinstance(func_args, list)
        assert self._execute_func is not None
        return self._execute_func(*func_args)

    def synchronize(self):
        pass

    def register_custom_class(self, obj):
        assert obj is not None
        return self
