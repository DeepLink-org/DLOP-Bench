from sys import float_repr_style
import pandas as pd
import matplotlib.pyplot as plt
import sys

data = pd.read_csv("conv2d_top10_flops.csv")
flops = [float(i) for i in data["Mflops"].values]
min = sys.float_info.max
max = -1.0
for i in flops:
    if i < min:
        min = i
    if i > max:
        max = i

min = 1
max = int(max) + 1
step = (max - min) / 25
print(min, max, step)
bins_list = [min+i*step for i in range(26)]
# print(bins_list)
fig, ax = plt.subplots(1, 1)
ax.hist(flops, bins=bins_list)
ax.set_title("Statistical Flops of Conv2d.")
ax.set_xticks(bins_list)
ax.set_xlabel("MFlops")
ax.set_ylabel("Freq of Occurrence.")
plt.xticks(rotation=45)
plt.savefig("./conv2d_top10_flops.png")
plt.show()

# print(flops)

# x = range(3)
# print(x)
# plt.bar(x, running_time)
# plt.xlabel("parameter configs")
# plt.ylabel("running time(ms)")
# plt.savefig("./conv2d_top3.png")
# plt.title("Performance of Conv2d")
# plt.xticks(x, ids)

# plt.show()



