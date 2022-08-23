import json
import pandas as pd
import matplotlib.pyplot as plt 

f = pd.read_csv("perf_result/softmax_perf.csv")
print(f)
#calc_cnt = [1, 2, 3, 4]
print(max(f["time_cost"].values[1:-1]))
print(min(f["time_cost"].values[1:-1]))
max_ = 0
maxs = []
mins = []
maxd = []
mind = []
min_ = 999

for i in range(len(f["time_cost"].values)): 
    if i == 0:
        continue
    cnt = f["time_cost"].values[i]
    if cnt > max_:
        maxs = f['item_0'][i]
        maxd = f['item_1'][i]
        max_ = cnt
    if cnt < min_:
        mins = f['item_0'][i]
        mind = f['item_1'][i]
        min_ = cnt
print(maxs)
print(maxd)
print(mins)
print(mind)


plt.hist(f["time_cost"].values[1:-1], bins = 20)
plt.show()


