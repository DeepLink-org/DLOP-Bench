from turtle import shape
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

data = pd.read_csv("../tmp/reshape.csv")
ids = data["id"]
networks = data["source_neural_network"].values
calls = data["num_of_calls"].values
calls_temp = data["num_of_calls"].values
running_time = [float(i)*1000 for i in data["running_time"].values]

# print(ids)
# print(networks)
# print(calls)
# print(running_time)

infos = data["parameters"].values
# print(len(infos))
# for info in infos:
#     # print(info)
# print(infos)
shapes = [eval(info)["input_shape"][0] for info in infos]
# print(shapes)

sizes = []
for shape in shapes:
    size = 1
    for num in shape:
        if num != 0:
            size *= num
    size *= 2
    size *= 4
    size /= (1024 * 1024)
    sizes.append(size)
# print(sizes)

n, bins, _ = plt.hist(sizes, bins=10, edgecolor='k', density=False)
for i in range(len(n)):
    plt.text((bins[i]+bins[i+1])/2, n[i]*1.01, int(n[i]), color = 'black', fontsize=10, horizontalalignment="center")
plt.xlabel('Mbytes')
plt.ylabel('Frequency of occurrence')
plt.savefig('../pictures/reshape_Mbytes.png')

calls = calls_temp
infos_set = []
calls_set1 = []
calls_set2 = []
times = []
for i in range(0, len(infos)):
    if infos[i] in infos_set:
        idx = infos_set.index(infos[i])
        calls_set1[idx] += calls[i]
        calls_set2[idx] += calls[i]
        times[idx] =(times[idx] + running_time[i])/2.0
    else:
        infos_set.append(infos[i])
        calls_set1.append(calls[i])
        calls_set2.append(calls[i])
        times.append(running_time[i])

# print(len(infos_set))
# print(len(calls_set1))
# print(len(times))

# print(calls_set1)
calls_set1, infos_set = zip(*sorted(zip(calls_set1, infos_set), reverse=True))
calls_set2, times = zip(*sorted(zip(calls_set2, times), reverse=True))

# print(calls_set1)
calls_top = calls_set1[:10]
times_top = times[:10]
# print(calls_top)
 
datas = [[1,2,3,4,5,1,2,3,4,5],
         [1,2,3,4,5,1,2,3,4,5],
         [1,2,3,4,5,1,2,3,4,5],
         [1,2,3,4,5,1,2,3,4,5]]

device_list = ['V100', 'A100', 'P100', 'T4']
labels = [f'#{i}' for i in range(10)]
x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
for i, (device, data) in enumerate(zip(device_list, datas)):
    ax.bar(x + (-len(datas) + 2 * i + 1) * width/2, data, width, label=device, edgecolor='black')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Performance (ms)')
ax.set_xlabel('Parameter configs')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
fig.tight_layout()
plt.show()
plt.savefig("../pictures/reshape_times.png")