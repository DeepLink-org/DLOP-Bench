# import imp
# import json
# import csv
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# import numpy as np

# with open("linear_all.json", 'r') as json_all:
#     data_json_all = json.load(json_all)
# item_info = ["item_0", "item_1", "item_2"]
# cnt_dict = {}
# for i in range(len(data_json_all["input"])):
#     key = []
#     key.append(tuple(data_json_all['input'][i]))
#     key.append(tuple(data_json_all['weight'][i]))
#     key.append(tuple(data_json_all['bias'][i]))
#     keyt = tuple(key)
#     if keyt in cnt_dict.keys():
#         cnt_dict[keyt] += 1
#     else:
#         cnt_dict[keyt] = 1    
# d_order=sorted(cnt_dict.items(),key=lambda x:x[1],reverse=True)
# d_order = d_order[:10]


# gpu_list = ['V100', 'A100', 'P100', 'T4']
# labels = [f'#{i}' for i in range(10)]
# datas = []
# for gpu in gpu_list:
#     perf_dir = "/mnt/lustre/yaofengchen/DLOP-Bench/perf_result"
#     perf_file = "linear_" + gpu + "_perf.csv"
#     perf_path = os.path.join(perf_dir, perf_file)
#     data = []
#     perf = pd.read_csv(perf_path)


# datas = [[1,2,3,4,5,1,2,3,4,5],
#          [1,2,3,4,5,1,2,3,4,5],
#          [1,2,3,4,5,1,2,3,4,5],
#          [1,2,3,4,5,1,2,3,4,5]]

# x = np.arange(len(labels))  # the label locations
# width = 0.2  # the width of the bars

# fig, ax = plt.subplots()
# for i, (device, data) in enumerate(zip(device_list, datas)):
#     ax.bar(x + (-len(datas) + 2 * i + 1) * width/2, data, width, label=device, edgecolor='black')

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Performance (ms)')
# # plt.suptitle('Performance Under Different Chips', x=0.55, y=.95, horizontalalignment='center')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()

# fig.tight_layout()

# plt.show()

