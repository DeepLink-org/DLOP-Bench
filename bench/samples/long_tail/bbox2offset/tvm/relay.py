# Copyright (c) OpenComputeLab. All Rights Reserved.

import onnx
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from gen_data import gen_np_args
import time

onnx_model = onnx.load("./bbox2offset.onnx")

x,  y = gen_np_args(3000, 4)

target = "cuda"

mod, params = relay.frontend.from_onnx(onnx_model)

with tvm.transform.PassContext(opt_level=1):
    executor = relay.build_module.create_executor(
        "graph", mod, tvm.cuda(0), target, params
    ).evaluate()

######################################################################
# Execute on TVM
# ---------------------------------------------
dtype = "float32"
time_start = time.time()
for i in range(1000):
    tvm_output = executor(tvm.nd.array(x.astype(dtype)), tvm.nd.array(y.astype(dtype))).numpy()
time_end = time.time()
print("bbox2offset tvm time cost: ", time_end - time_start)
