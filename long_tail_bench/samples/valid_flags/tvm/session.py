import onnx
import onnxruntime
import time


onnx_model = onnx.load_model("./valid_flags.onnx")
session = onnxruntime.InferenceSession(onnx_model.SerializeToString())

input = {}

time_start = time.time()
for i in range(1000):
    output = session.run(None, input)
time_end = time.time()
print("onnxruntime time cost: ", time_end - time_start)