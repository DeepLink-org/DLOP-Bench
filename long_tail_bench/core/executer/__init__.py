from long_tail_bench.common import FrameType, FRAMEWORK

if FRAMEWORK is FrameType.Parrots:
    from .parrots_executer import (
        ParrotsExecuter as Executer,
        log_debug_info,
        tensor_type,
        trans_tensor_to_np,
        set_runtime_exec_mode,
        get_runtime_exec_mode,
    )
elif FRAMEWORK is FrameType.Torch:
    from .torch_executer import (
        TorchExecuter as Executer,
        log_debug_info,
        tensor_type,
        trans_tensor_to_np,
        set_runtime_exec_mode,
        get_runtime_exec_mode,
    )
elif FRAMEWORK is FrameType.XLA:
    from .xla_executer import (  # noqa
        XLAExecuter as Executer,
        log_debug_info,
        tensor_type,
        trans_tensor_to_np,
        set_runtime_exec_mode,
        get_runtime_exec_mode,
    )
elif FRAMEWORK is FrameType.JAX:
    from .jax_executer import (  # noqa
        JAXExecuter as Executer, 
        log_debug_info, 
        tensor_type,
        trans_tensor_to_np,
        set_runtime_exec_mode,
        get_runtime_exec_mode,
    )

__all__ = (
    [
        "Executer",
        "log_debug_info",
        "tensor_type",
        "trans_tensor_to_np",
        "set_runtime_exec_mode",
        "get_runtime_exec_mode",
    ]
    if FRAMEWORK
    else []
)
