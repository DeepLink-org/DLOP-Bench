# Long Tail Bench
````````````````````````````````````````````````````````````````````````````````````````
针对算子性能报告更改/新增了如下文件:
    DLOP-Bench/run.sh:测试算子的运行脚本，根据需要直接解掉对应注释即可,涉及到的算子均在脚本中；

    DLOP-Bench/op_csv:其中包含了每个对应算子的参数信息和调用信息;

    DLOP-Benc/tmp:其中保存每个对应算子的性能测试结果;

    DLOP-Bench/scripts:其中包含了每个对应算子的画图脚本，从DLOP-Benc/tmp中读取对应算子的测试结果，生成相应图片在DLOP-Benc/pictures中;

    DLOP-Bench/pictures:其中保存每个对应算子所生成的图片;

    同时，更新了每个算子测试时所用到的包含算子参数信息的json文件,从parameter仓库获取。

    测试流程:
        根据算子更改运行脚本；

        根据算子信息更改DLOP-Bench/long_tail_bench/core/engine.py中函数new_make_data中的参数个数信息，
        以及函数perf_per_case中readCSV的路径名(对应DLOP-Bench/op_csv中的csv),
        以及writeCSV的路径名(对应DLOP-Benc/tmp文件夹,文件名按算子自定义)；

        运行脚本run.sh,会在DLOP-Benc/tmp中生成结果；

        根据算子执行DLOP-Bench/scripts中的画图脚本，在DLOP-Bench/pictures中生成对应结果图片。
````````````````````````````````````````````````````````````````````````````````````````

要在不同机器上测试不同算子，针对每一个算子只需要更改四个地方。
目前是在实验室S集群上运行cat算子的状态，以将其更改为index_select算子为例：
1：首先更改运行脚本WAIC/DLOP-Bench/run.sh，将cat对应的行注释掉，再将index_select对应的行解掉注释；

2: 依据DLOP-Bench/long_tail_bench/samples/basic/index_select/__init__.py:line20:requires_grad=[False] * 3中的数字3，将DLOP-Bench/long_tail_bench/core/engine.py:line184:args = executer.generate_args(args_cases[0], [False]*2, np_args_generator)中的数字2替换为3，不同算子依赖不同的DLOP-Bench/long_tail_bench/samples/basic/op_name/__init__.py；

3：更改DLOP-Bench/long_tail_bench/core/engine.py:line298:with open("/mnt/lustre/zhoushenglong/WAIC/DLOP-Bench/op_csv/cat.csv") as readCSV:中的文件名cat.csv为index_select.csv，此处是为了读取不同算子的相关数据，与机器型号无关，直接改文件名即可；

4: 更改DLOP-Bench/long_tail_bench/core/engine.py:line300:with open("./tmp/cat_A100.csv", "w") as writeCSV:中的文件名cat_A100为index_select_A100，此处是为了生成不同算子的结果，A100为具体的机器型号，如机器为V100，则应为cat_V100到index_select_V100，一定要根据具体机器进行修改，否则不同机器生成的结果会被覆盖掉。

note：涉及到路径的地方均为相对路径，只需要根据自己的机器更改为绝对路径即可。

````````````````````````````````````````````````````````````````````````````````````````

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