from abc import abstractmethod, ABC


class AbstractExecuter(ABC):
    """Abstract class of backend executer.
    """
    @abstractmethod
    def generate_args(self, case, requires_grad):
        """Generate sample function execution args."""
        pass

    @abstractmethod
    def prepare(self, stage_mode):
        """Prepare sample func to execute."""
        pass

    @abstractmethod
    def clone_func_args(self, func_args):
        """Clone sample function args."""
        pass

    @abstractmethod
    def execute(self, func_args):
        """Execute sample function."""
        pass

    @abstractmethod
    def synchronize(self):
        """Do sync"""
        pass

    @abstractmethod
    def assert_correct(self, val_a, val_b):
        """Check outputs."""
        pass

    @abstractmethod
    def backward(self, ret_t, func_arg):
        """Do backward of sample function outputs."""
        pass

    @abstractmethod
    def reset_grad(self, func_args, requires_grad):
        """Reset grads of sample function inputs."""
        pass

    @abstractmethod
    def save_timeline_start(self):
        """Start record timeline."""
        pass

    @abstractmethod
    def save_timeline_end(self):
        """End record timeline."""
        pass

    @abstractmethod
    def register_custom_class(self, obj):
        """Register custom class."""
        pass


class BaseCaseExecuter(AbstractExecuter):
    """Base class of backend executer.

    Args:
        core_func(Function): Sample origin execution function.
        args_adaptor(Function): Sample args adaptor function, 
            it transform numpy inputs to tensor.
    """
    def __init__(self, core_func, args_adaptor):
        self._origin_func = core_func
        self._args_adaptor = args_adaptor
        self._execute_func = None

    def adapt_args(self, np_args):
        """Transform sample numpy args to tensor.
        Args:
            np_args(list[numpy.ndarray]): Sample numpy args.
        Returns:
            tensor: tensor transformed by backend.
        """
        assert isinstance(np_args, list)
        return self._args_adaptor(np_args)

    def execute(self, func_args):
        """Execute sample execution function 
        Args:
            func_args(list[tensor]): Sample tensor args.
        Returns:
            list[Any]: Sample execution function outputs.
        """
        assert isinstance(func_args, list)
        assert self._execute_func is not None
        return self._execute_func(*func_args)

    def synchronize(self):
        pass

    def register_custom_class(self, obj):
        """Register custom class."""
        assert obj is not None
        return self
