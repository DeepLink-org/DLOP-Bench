import onnx
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from gen_data import gen_np_args
import time

onnx_model = onnx.load("./iou_loss.onnx")


pred, target_args = gen_np_args(128)

target = "cuda"

mod, params =relay.frontend.from_onnx(onnx_model)

with tvm.transform.PassContext(opt_level=1):
    executor = relay.build_module.create_executor(
            "graph", mod, tvm.cuda(0), target, params).evaluate()


dtype = "float32"
time_start = time.time()
for i in range(1000):
    tvm_output = executor(tvm.nd.array(pred.astype(dtype)),\
        tvm.nd.array(target_args.astype(dtype))).numpy()
time_end = time.time()
print("iou_loss tvm time cost: ", time_end - time_start)
