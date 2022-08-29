# Copyright (c) OpenComputeLab. All Rights Reserved.
import onnx
import onnxruntime
import time
from gen_data import gen_np_args


onnx_model_path = "./"
onnx_model = onnx.load_model(onnx_model_path + "box_area.onnx")
session = onnxruntime.InferenceSession(onnx_model.SerializeToString())

x = gen_np_args(20, 5)
input = {session.get_inputs()[0].name:x}

time_start = time.time()
for i in range(1000):
    output = session.run(None, input)
time_end = time.time()
print("onnxruntime time cost: ", time_end - time_start)