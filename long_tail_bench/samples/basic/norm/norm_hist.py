import json
import matplotlib.pyplot as plt 

with open("norm_all.json", 'r') as f:
    data = json.load(f)
calc_cnt = []
max_ = 0
maxs = []
mins = []
maxd = []
mind = []
min_ = 999
for i in range(len(data['x1'])):
    cnt = 1
    for d in range(len(data['x1'][i])):
        cnt = cnt * data['x1'][i][d]
    cnt = cnt * 4
    if cnt > max_:
        maxs = data['x1'][i]
        maxd = data['x2'][i]
        max_ = cnt
    if cnt < min_:
        mins = data['x1'][i]
        mind = data['x2'][i]
        min_ = cnt
    calc_cnt.append(cnt / (10**6))
#calc_cnt = [1, 2, 3, 4]
n, bins, _ = plt.hist(calc_cnt, bins=10, edgecolor='k', density=False)
for i in range(len(n)):
    plt.text((bins[i]+bins[i+1])/2, n[i]*1.01, int(n[i]), color = 'black', fontsize=10, horizontalalignment="center")

plt.xlabel('Mbytes')
plt.ylabel('Frequency of occurrence')
# plt.yscale("log", base=10)
# plt.text()
# plt.show()
plt.savefig('norm_top10_mem.pdf')
plt.show()