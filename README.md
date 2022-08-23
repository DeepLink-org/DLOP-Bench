# update
一些op下面添加了两个script
topk.py： 通过json统计出出现次数最多的topk config
flops_cnt.py : 生成flops/memory access的直方图






# Long Tail Bench

A benchmark for long tail operators in deep learning. Deep neural networks have brought significant innovations in many domains, such as computer vision, natural language processing, and speech recognition. To this end, many new operators have also been developed to attain better accuracy in the specific domain, such as the operators about anchors in object detection and operators about agent-environment interaction in reinforcement learning. We name the new operator which has no corresponding implementation in the device compute library and has to be composed of meta-operations and control flow in python interpreter as long-tail operators. This is inspired by the meaning of long tail phenomenon in business and statistics that products with small sales but many types, not valued originally, in fact, have a huge total amount. Benchmark suite is the quantitative foundation for the improvement of related research and industry, which plays a very important role in pushing technology development. Unfortunately, there have been no representative benchmark suites that can present the importance of long-tail operators and help guide to switch the focus of compiler researchers and chip vendors from ordinary neural network operators to them. LongTail-Bench is proposed to fill the gap, which can help to evaluate the existing deep learning systems from underlying hardware up to the algorithm, and further guide the research directions of the algorithm, compiler, and chip researchers.

## Highlights
- **Benchmark:** focus on compiling long tail operators
- **Execution framework:** The main body is an execution engine, compatible with different modes, such as eager and different compiling mode
- **100+ samples:** has collected 100+ long tail samples from different deep learning models, more than half of them come from [open-mmlab](https://github.com/open-mmlab), see [samples](long_tail_bench/samples/README.md) for more detail
- **Rich sample features:** such as source repo, url and semantic tags, if you want to get detail info, use the bash command as follows:
```bash
# show features of the running samples, including sample source, url and semantic tags
FRAMEWORK=parrots python ./long_tail_bench/api/api.py -c bbox2delta -sc
```
- **CV related strongly** 
- **TorchScript and XLA mode supported:** some samples have torch script and xla implementation codes, and the execution framework is also able to run it if you could provide corresponding environment. See the instruction
- **Tools:** grip samples conveniently on your own

## Getting Started Instruction
This is parrots running command demo that illustrates how you can use LongTail-Bench to test samples performance. These apis can also be used in backend torch or xla, just set corresponding FRAMEWORK environment:
```bash
# prepare parrots environmant
source pat_latest
# config long tail bench PYTHONPATH
cd LongTail-Bench
export PYTHONPATH=./long_tail_bench:$PYTHONPATH
# run one sample
FRAMEWORK=parrots python ./long_tail_bench/api/api.py -c tblr2bbox
# run several samples
FRAMEWORK=parrots python ./long_tail_bench/api/api.py -c tblr2bbox,bbox2delta
# run all samples
FRAMEWORK=parrots python ./long_tail_bench/api/api.py
# run several stages, 1: eager stage, 2: fixed shape jit stage, 3: fixed shape coder stage,
# 4: dynamic shape jit stage, 5: dynamic shape coder stage, 2、3、4 just for parrots compiler
FRAMEWORK=parrots python ./long_tail_bench/api/api.py -st 1,2,3
# show all tags
FRAMEWORK=parrots python ./long_tail_bench/api/api.py -sg
# show samples of one tag
FRAMEWORK=parrots python ./long_tail_bench/api/api.py -sot AdvancedIndexing
# show features of the running samples, including sample source, url and semantic tags
FRAMEWORK=parrots python ./long_tail_bench/api/api.py -c bbox2delta -sc
# transform results recorded to excel, it will create csv or xlsx file in directory ./long_tail_bench/results/
FRAMEWORK=parrots python ./long_tail_bench/api/export_result_to_excel.py
# parrots PAT debug
FRAMEWORK=parrots PARROTS_PAT_DEBUG=1 python ./long_tail_bench/api/api.py
# benchmark debug
FRAMEWORK=parrots BENCH_DEBUG=1 python ./long_tail_bench/api/api.py

```

If you want to test sample performance using torch backend, you can see the demo as follows:
```bash
# prepare pytorch environment, torch 1.3 best
...
# config long tail bench PYTHONPATH
cd LongTail-Bench
export PYTHONPATH=./long_tail_bench:$PYTHONPATH
# run parrots sample implementation using torch backend in that torch samples implementation are equal to parrots' nearly
FRAMEWORK=torch SAMPLE_IMPL=parrots srun -p pat_dev --gres=gpu:1 python ./long_tail_bench/api/api.py -st 1
```

XLA running demo as follows:
```bash
# prepare xla environment
...
# config long tail bench PYTHONPATH
cd LongTail-Bench
export PYTHONPATH=./long_tail_bench:$PYTHONPATH
# run xla samples
FRAMEWORK=xla TF_XLA_FLAGS=--tf_xla_auto_jit=2 XLA_FLAGS=--xla_gpu_cuda_data_dir=.../cuda-10.1 python ./long_tail_bench/api/api.py -st 1
```

Sample gripper usage, it will grip the sample decorated and save files generated to directory ./long_tail_bench/samples/:
```python
>>>from long_tail_bench.tools.sample_gripper import grip
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
