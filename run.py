import os

op_list = ["linear", "bmm", "ger", "mv", "randperm", "norm", "einsum", "topk", "sort"]

for op_name in op_list:
    run_command = "srun -p caif_debug --gres=gpu:1 bash run.sh {}".format(op_name)
    os.system(run_command)