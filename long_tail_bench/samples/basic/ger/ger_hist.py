import json
import matplotlib.pyplot as plt 

with open("ger_all.json", 'r') as f:
    data = json.load(f)
calc_cnt = []
max_ = 0
maxs = []
mins = []
maxd = []
mind = []
min_ = 999
for i in range(len(data['input'])):
    cnt = data['input'][i][0] * data['vec2'][i][0]
    if cnt > max_:
        maxs = data['input'][i]
        maxd = data['vec2'][i]
        max_ = cnt
    if cnt < min_:
        mins = data['input'][i]
        mind = data['vec2'][i]
        min_ = cnt
    calc_cnt.append(cnt / 1000)

n, bins, _ = plt.hist(calc_cnt, bins=10, edgecolor='k', density=False)
for i in range(len(n)):
    plt.text((bins[i]+bins[i+1])/2, n[i]*1.01, int(n[i]), color = 'black', fontsize=10, horizontalalignment="center")

plt.xlabel('KFLOPs')
plt.ylabel('Frequency of occurrence')
# plt.yscale("log", base=10)
# plt.text()
# plt.show()
plt.savefig('ger_top10_flops.pdf')
plt.show()