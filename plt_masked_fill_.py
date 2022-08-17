import pandas as pd
import matplotlib.pyplot as plt
import sys

data = pd.read_csv("masked_fill_.csv")
ids = data["id"]
networks = data["source_neural_network"].values
calls = data["num_of_calls"].values
calls_temp = data["num_of_calls"].values
running_time = [float(i)*1000 for i in data["running_time"].values]

# print(ids)
# print(networks)
# print(calls)
# print(running_time)

# sum_calls = sum(calls)
# print(sum_calls)
# freps = ["%.2f" % (call/sum_calls*100) for call in calls]
# freps = [float(frep) for frep in freps]
# print(freps)

infos = data["parameters"].values
# print(len(infos))
# for info in infos:
#     print(info)
# print(infos)
shapes = [eval(info)["input"][0] for info in infos]
# indexs = [eval(info)["index"][0] for info in infos]
# print(shapes)
# print(indexs)

sizes = []
for i in range(0, len(shapes)):
    size = 1
    for j in range(0,len(shapes[i])):
        size *= shapes[i][j]
    sizes.append(size)

flops = []
for i in range(0, len(running_time)):
    # print(sizes[i])
    # print(running_time[i])
    flop = sizes[i]/running_time[i]
    flop /= 1000000
    flop = flop * 1.0
    flops.append(flop)

# print(flops)
# print(calls)
flops, calls = zip(*sorted(zip(flops, calls), reverse=False))
# print(flops)
# print(calls)

# min = sys.float_info.max
# max = -1.0
# for flop in flops:
#     if flop < min:
#         min = flop  
#     if flop > max:
#         max = flop
# max = int(max) + 1
# step = (max - min) / 25
# print(min, max, step)
# bins_list = [int(min+i*step) for i in range(26)]


x = range(len(flops))
plt.bar(x, calls)
plt.title("Statistical flops of masked_fill_")
plt.xlabel("MFlops")
plt.ylabel("Count")
plt.savefig("./masked_fill__flops.png")
# y = range(26)
# print(bins_list)
# plt.xticks(y, bins_list)
plt.show()

infos_set = []
calls_set1 = []
calls_set2 = []
times = []
for i in range(0, len(infos)):
    if infos[i] in infos_set:
        idx = infos_set.index(infos[i])
        calls_set1[idx] += calls_temp[i]
        calls_set2[idx] += calls_temp[i]
        times[idx] =(times[idx] + running_time[i])/2.0
    else:
        infos_set.append(infos[i])
        calls_set1.append(calls_temp[i])
        calls_set2.append(calls_temp[i])
        times.append(running_time[i])

print(len(infos_set))
# print(len(calls_set1))
# print(len(times))

calls_set1, infos_set = zip(*sorted(zip(calls_set1, infos_set), reverse=True))
calls_set2, times = zip(*sorted(zip(calls_set2, times), reverse=True))

# print(calls_set1)
calls_top = calls_set1[:10]
times_top = times[:10]
# print(calls_top)
 
# x = range(len(calls_top))
# plt.bar(x, times_top)
# plt.title("Performance of masked_fill_")
# plt.xlabel("parameter configs")
# plt.ylabel("running time(ms)")
# plt.savefig("./masked_fill__times.png")
# plt.xticks(x, calls_top)
# plt.show()
