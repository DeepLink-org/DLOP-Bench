## Introduction

DLOP-Bench is an open source benchmark suite for deep learning operators. It has the following three major features:

- **Operators at the deep learning framework level**


We focus on the operator at the deep learning framework level (such as torch.convolution) and do not dive into the implementation details of each operator (implicit gemm implementation or winograd implementation and the related algorithm selection). One can easily benchmark the operators on a certain AI accelerator as long as they finish the adaption on a deep learning framework.

- **Basic Operators and Domain-specific long-tail operators**


Besides basic operators like convolution, pooling, and normalization, we also collect many representative domain-specific operators mainly from object detection, instance segmentation, and other computer vision directions in [OpenMMLab](https://github.com/open-mmlab). These operators have no dedicated implementation on deep learning accelerators and have to resort to the Python interpreter. As such, they will always be broken down into large numbers of basic operators. They incur a lot of function calls, as well as data transfer and context switching costs. We name them long-tail operators.

- **Benchmarking deep learning accelerators, frameworks and compilers**


From the operator level, this benchmark suite can provide a more microscopic assessment from multiple aspects, including accelerator hardware specifications, deep learning frameworks and deep learning compilers.

## Highlights

- **Execution framework.** The main body is an execution engine, compatible with different deep learning frameworks (PyTorch, TensorFlow, JAX, and so on) with different execution modes, such as eager and graph mode.
- **200+ basic operators.** We collected the operators from models in [OpenMMLab](https://github.com/open-mmlab). The input information consists of two parts: input tensor shape and attributes information. We run the models and record input configurations of each operator. For each input configuration, we save them in CSV format for evaluation.
- **100+ long-tail samples.** It has collected 100+ long tail samples from different deep learning models with representative syntax features, mainly from [OpenMMLab](https://github.com/open-mmlab), see [samples](bench/samples/README.md) for more detail.

## Getting Started Instruction


First, download the latest source code:
```bash
git clone git@github.com:OpenComputeLab/DLOP-Bench.git
```

To show the structure of source code, we can use the following command:
```bash
cd DLOP-Bench
tree -d -L 1 ./bench
```
The implementation functions of basic and long tail operators are located in ./bench/samples/.


Here is a command demo that illustrates how you can use DLOP-Bench to test samples performance.

```bash
# config bench PYTHONPATH
cd bench
export PYTHONPATH=./bench:$PYTHONPATH
If you want to test sample performance using torch backend, you can see the demo as follows:
```bash
# prepare pytorch environment, python 3 & torch 1.10 or 1.12 best
pip3 install torch 
# run the operator tblr2bbox using torch backend in eager mode
FRAMEWORK=torch python ./bench/api/api.py -c aeloss -st 1


# run one sample
FRAMEWORK=torch python ./bench/api/api.py -c tblr2bbox
# run several samples
FRAMEWORK=torch python ./bench/api/api.py -c tblr2bbox,bbox2delta
# run all samples
FRAMEWORK=torch python ./bench/api/api.py
# run several stages, 1: eager stage, 2: fixed shape jit stage, 3: fixed shape coder stage,
# 4: dynamic shape jit stage, 5: dynamic shape coder stage, 
FRAMEWORK=torch python ./bench/api/api.py -st 1,2,3
# show all tags
FRAMEWORK=torch python ./bench/api/api.py -sg
# show samples of one tag
FRAMEWORK=torch python ./bench/api/api.py -sot AdvancedIndexing
# show features of the running samples, including sample source, url and semantic tags
FRAMEWORK=torch python ./bench/api/api.py -c bbox2delta -sc
# transform results recorded to excel, it will create csv or xlsx file in directory ./bench/results/
FRAMEWORK=torch python ./bench/api/export_result_to_excel.py
# benchmark debug
FRAMEWORK=torch BENCH_DEBUG=1 python ./bench/api/api.py

```
These apis can also be used in backend torch, tensorflow, or xla, just set corresponding FRAMEWORK environment:
If you want to test sample performance using tensorflow, or XLA backend, you can see the demo as follows:


XLA running demo as follows:

```bash
# prepare xla environment
...
# config bench PYTHONPATH
cd bench
export PYTHONPATH=./bench:$PYTHONPATH
# run xla samples
FRAMEWORK=xla TF_XLA_FLAGS=--tf_xla_auto_jit=2 XLA_FLAGS=--xla_gpu_cuda_data_dir=.../cuda-10.1 python ./bench/api/api.py -st 1
```

Sample gripper usage, it will grip the sample decorated and save files generated to directory ./bench/samples/:

```python
>>>from bench.tools.sample_gripper import grip
>>>@grip(class_entry="forward")
...class Test(obj):
...    def __init__(self):
...        pass
...
...
...    def forward(self):
...        pass

>>>@grip
...def bbox2delta(a, b, c):
...    pass
...
```

## Paper

[WIP] LongTail-Bench: A Benchmark for long tail operators in Deep Learning
