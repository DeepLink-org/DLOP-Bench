import json
import matplotlib.pyplot as plt 

with open("mv_all.json", 'r') as f:
    data = json.load(f)
calc_cnt = []
max_ = 0
maxs = []
mins = []
maxd = []
mind = []
min_ = 999
for i in range(len(data['x1'])):
    cnt = data['x1'][i][0] * data['x1'][i][1] * 2
    if cnt > max_:
        maxs = data['x1'][i]
        maxd = data['x2'][i]
        max_ = cnt
    if cnt < min_:
        mins = data['x1'][i]
        mind = data['x2'][i]
        min_ = cnt
    calc_cnt.append(cnt / (10**6))
n, bins, _ = plt.hist(calc_cnt, bins=10, edgecolor='k', density=False)
for i in range(len(n)):
    plt.text((bins[i]+bins[i+1])/2, n[i]*1.01, int(n[i]), color = 'black', fontsize=10, horizontalalignment="center")

plt.xlabel('MFLOPs')
plt.ylabel('Frequency of occurrence')
# plt.yscale("log", base=10)
# plt.text()
# plt.show()
plt.savefig('mv_top10_flops.pdf')
plt.show()