import json
from turtle import shape
import matplotlib.pyplot as plt 

with open("einsum_all.json", 'r') as f:
    data = json.load(f)
calc_cnt = []
max_ = 0
min_ = 999
for i in range(len(data['x1'])):
    cnt_1 = 1
    cnt_2 = 1
    cnt_3 = 1
    shape_in = {}
    s_in = ""
    s_out_tmp = ""
    s_out = ""
    for item in data['x1'][i]:
        if 'a' <= item <= 'z' or item == '/':
            s_in += item
        elif item == '>':
            s_out_tmp = data['x1'][i][data['x1'][i].find(item) + 1:]
            break
    for item in s_out_tmp:
        if 'a' <= item <= 'z':
            s_out += item
    s_in = s_in.split('/')
    for j in range(len(s_in[0])):
        shape_in[s_in[0][j]] = data['x2'][i][j]
    for j in range(len(s_in[1])):
        shape_in[s_in[1][j]] = data['x3'][i][j]
    for item in s_out:
        cnt_3 = cnt_3 * shape_in[item]
    for item in data['x2'][i]:
        cnt_1 *= int(item)
    for item in data['x3'][i]:
        cnt_2 *= int(item)
    cnt = (cnt_1 + cnt_2 + cnt_3) * 4
    if cnt > max_:
        max_ = cnt
    if cnt < min_:
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
plt.savefig('einsum_top10_mem.pdf')