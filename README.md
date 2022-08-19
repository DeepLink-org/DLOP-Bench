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
在不同的gpu上测试的话，有两个地方需要改：
- run.py中的run_command变量为运行任务命令
- run.py中的gpu_name变量为gpu的名字，方便为不同gpu上测试的结果文件命名
```
python run.py
```