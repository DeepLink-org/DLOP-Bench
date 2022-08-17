import json
from turtle import shape
import matplotlib.pyplot as plt 

with open("einsum_all.json", 'r') as f:
    data = json.load(f)
calc_cnt = []
max_ = 0
min_ = 999
for i in range(len(data['x1'])):
    cnt = 1
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
        cnt = cnt * shape_in[item]
    cnt *= 2

    if cnt > max_:
        max_ = cnt
    if cnt < min_:
        min_ = cnt
    calc_cnt.append(cnt)
#calc_cnt = [1, 2, 3, 4]
gn = int((max(calc_cnt) - min(calc_cnt))/10000)
print(max(calc_cnt))
print(min(calc_cnt))


plt.hist(calc_cnt, bins = 10)
plt.xlabel("flops")
plt.ylabel("count")
plt.savefig("einsum_count.png")
# plt.text()
plt.show()