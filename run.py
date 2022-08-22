import os

op_list = ["linear", "bmm", "ger", "mv", "rand", "randperm", "norm", "einsum", "normalize", "topk", "sort"]
gpu_name = "V100"

for op_name in op_list:
    # test perf
    run_command = "srun -p caif_debug --gres=gpu:1 bash run.sh {}".format(op_name)
    os.system(run_command)
    
    # rename perf csv
    dir_name = "perf_result"
    old_name = op_name + "_perf.csv"
    new_name = op_name + "_" + gpu_name + "_perf.csv"
    rename_command = "cd {} && mv {} {} && cd ..".format(dir_name, old_name, new_name)
    os.system(rename_command)

    # rename profiler txt
    dir_name = "profiler_result"
    old_name = op_name + "_profiler.txt"
    new_name = op_name + "_" + gpu_name + "_profiler.txt"
    rename_command = "cd {} && mv {} {} && cd ..".format(dir_name, old_name, new_name)
    os.system(rename_command)