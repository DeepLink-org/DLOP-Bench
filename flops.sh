cat conv2d_top10_profiler.txt | grep "aten::conv2d" | awk '{print $NF}' > conv2d_top10_flops.txt
