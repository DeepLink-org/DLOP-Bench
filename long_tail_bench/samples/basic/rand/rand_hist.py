import json
import matplotlib.pyplot as plt 

with open("rand_all.json", 'r') as f:
    data = json.load(f)
calc_cnt = []
max_ = 0
min_ = 999
for i in range(len(data['size'])):
    cnt = 1
    if isinstance(data['size'][i], int):
        cnt = data['size'][i]
    else:
        for item in data['size'][i]:
            cnt = cnt * item
    if cnt > max_:
        max_ = cnt
    if cnt < min_:
        min_ = cnt
    calc_cnt.append(cnt * 4 / (10**6))
#calc_cnt = [1, 2, 3, 4]
n, bins, _ = plt.hist(calc_cnt, bins=10, edgecolor='k', density=False)
for i in range(len(n)):
    plt.text((bins[i]+bins[i+1])/2, n[i]*1.01, int(n[i]), color = 'black', fontsize=10, horizontalalignment="center")

plt.xlabel('Mbytes')
plt.ylabel('Frequency of occurrence')
# plt.yscale("log", base=10)
# plt.text()
# plt.show()
plt.savefig('rand_top10_mem.pdf')
plt.show()