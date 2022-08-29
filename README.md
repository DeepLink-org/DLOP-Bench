## Introduction

DLOP-Bench is an open source benchmark suite for deep learning operators. It has the following three major features:

- **Operators at the deep learning framework level**


We focus on the operator at the deep learning framework level (such as torch.convolution) and do not dive into the implementation details of each operator (implicit gemm implementation or winograd implementation and the related algorithm selection). One can easily benchmark the operators on a certain AI accelerator as long as they finish the adaption on a deep learning framework.

- **Basic operators and domain-specific long-tail operators**


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
git clone https://github.com/OpenComputeLab/DLOP-Bench.git
```

To show the structure of source code, we can use the following command:
```bash
cd DLOP-Bench
tree -d -L 1 ./bench
```
The implementation functions of basic and long tail operators are located in ./bench/samples/.

### Basic Operators

Here is a command demo that illustrates how you can use DLOP-Bench to test basic operators.

```bash
# config bench PYTHONPATH
cd DLOP-bench
export PYTHONPATH=./bench:$PYTHONPATH
If you want to test sample performance using torch backend, you can see the demo as follows:
```bash
# prepare pytorch environment, python 3 & torch 1.10 or 1.12 best
pip install torch 
# run the operator abs using torch backend
FRAMEWORK=torch python ./bench/api/api.py -c abs

# get the usage information
python ./bench/api/api.py --help
```

### Long-tail Operators

From long-tail operators, this benchmark suite provides several stages to test their performance as below: 
- **stage 1** : eager mode.
- **stage 3** : graph mode with fixed shape jit.
- **stage 5** : graph mode with dynamic shape jit.

This benchmark suite supports the execution of all long-tail operators in stage 1, while some operators fail to run in stage 3 and 5 because they are unsupported in the given deep learning compiler.
Here is a command demo to test long-tail operators.

```bash
# config bench PYTHONPATH
cd bench
export PYTHONPATH=./bench:$PYTHONPATH
# run the operator aeloss using torch backend in eager mode
FRAMEWORK=torch python ./bench/api/api.py -c aeloss -st 1


# run one sample
FRAMEWORK=torch python ./bench/api/api.py -c tblr2bbox
# run several samples
FRAMEWORK=torch python ./bench/api/api.py -c tblr2bbox,bbox2delta
# run all samples
FRAMEWORK=torch python ./bench/api/api.py
# run several stages, 1: eager stage, 3: fixed shape coder stage, 5: dynamic shape coder stage, 
FRAMEWORK=torch python ./bench/api/api.py -st 1,3,5
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
These apis can also be used in backend torch, tensorflow, or xla, just set corresponding FRAMEWORK environment.
While all the operators can be tested using torch backend, some operators may raise an AssertionError in other backends if their corresponding implementation codes have not been added yet.
You can wait for our update or add the codes yourself.

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

## How to add new operator

- Create a folder named after the operator in the ``./bench/samples/basic`` directory
- Copy the json file of the operator parameter information table generated by the operator acquisition module into the folder
- Create ``__init__.py`` and ``torch_impl.py`` files, if you need to test other framework operators, you can refer to ``torch_impl.py``
In ``__init__.py``, you need to implement two functions ``get_sample_config`` and ``gen_np_args``, and then register the two functions using ``register_sample``.
In ``torch_impl.py`` you need to implement the function ``args_adaptor``, which perform data preparation and the operator definition you are going to add. Then, ``executor_creator`` function is needed to register the above two functions into the benchmark.
