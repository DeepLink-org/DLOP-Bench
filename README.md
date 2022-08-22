## 性能测试的算子
### BLAS
linear  bmm ger mv
### Distribution
rand    randperm
### Linalg
norm    einsum  normalize
### Sort
topk    sort


## 性能测试脚本
在不同的gpu上测试，有两个地方需要改：
- run.py中的gpu_name，为不同gpu上测试的结果文件命名。测试结果在perf_result和profiler_result
- run.py中的run_command变量为运行任务命令
```
python run.py
```
