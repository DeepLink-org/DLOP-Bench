# Copyright (c) OpenComputeLab. All Rights Reserved.
import onnx
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay

onnx_model = onnx.load("./valid_flags.onnx")

target = "cuda"

mod, params =relay.frontend.from_onnx(onnx_model)

with tvm.transform.PassContext(opt_level=1):
    executor = relay.build_module.create_executor(
            "graph", mod, tvm.cuda(0), target, params).evaluate()


dtype = "float32"
tvm_output = executor().numpy()
