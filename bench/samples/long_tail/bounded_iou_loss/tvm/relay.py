# Copyright (c) OpenComputeLab. All Rights Reserved.
import onnx
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from gen_data import gen_np_args

onnx_model = onnx.load("./bounded_iou_loss.onnx")


pred, target_args = gen_np_args(4)

target = "cuda"

mod, params =relay.frontend.from_onnx(onnx_model)

with tvm.transform.PassContext(opt_level=1):
    executor = relay.build_module.create_executor(
            "graph", mod, tvm.gpu(0), target, params).evaluate()


dtype = "float32"
tvm_output = executor(tvm.nd.array(pred.astype(dtype)),\
        tvm.nd.array(target_args.astype(dtype))).numpy()
