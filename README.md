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

This is parrots running command demo that illustrates how you can use LongTail-Bench to test samples performance. These apis can also be used in backend torch or xla, just set corresponding FRAMEWORK environment:

```bash
# prepare parrots environmant
source pat_latest
# config long tail bench PYTHONPATH
cd LongTail-Bench
export PYTHONPATH=./bench:$PYTHONPATH
# run one sample
FRAMEWORK=parrots python ./bench/api/api.py -c tblr2bbox
# run several samples
FRAMEWORK=parrots python ./bench/api/api.py -c tblr2bbox,bbox2delta
# run all samples
FRAMEWORK=parrots python ./bench/api/api.py
# run several stages, 1: eager stage, 2: fixed shape jit stage, 3: fixed shape coder stage,
# 4: dynamic shape jit stage, 5: dynamic shape coder stage, 2、3、4 just for parrots compiler
FRAMEWORK=parrots python ./bench/api/api.py -st 1,2,3
# show all tags
FRAMEWORK=parrots python ./bench/api/api.py -sg
# show samples of one tag
FRAMEWORK=parrots python ./bench/api/api.py -sot AdvancedIndexing
# show features of the running samples, including sample source, url and semantic tags
FRAMEWORK=parrots python ./bench/api/api.py -c bbox2delta -sc
# transform results recorded to excel, it will create csv or xlsx file in directory ./bench/results/
FRAMEWORK=parrots python ./bench/api/export_result_to_excel.py
# parrots PAT debug
FRAMEWORK=parrots PARROTS_PAT_DEBUG=1 python ./bench/api/api.py
# benchmark debug
FRAMEWORK=parrots BENCH_DEBUG=1 python ./bench/api/api.py

```

If you want to test sample performance using torch backend, you can see the demo as follows:

```bash
# prepare pytorch environment, torch 1.3 best
...
# config long tail bench PYTHONPATH
cd LongTail-Bench
export PYTHONPATH=./bench:$PYTHONPATH
# run parrots sample implementation using torch backend in that torch samples implementation are equal to parrots' nearly
FRAMEWORK=torch SAMPLE_IMPL=parrots srun -p pat_dev --gres=gpu:1 python ./bench/api/api.py -st 1
```

XLA running demo as follows:

```bash
# prepare xla environment
...
# config long tail bench PYTHONPATH
cd LongTail-Bench
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
