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
    cnt = cnt * 3
    if cnt > max_:
        maxs = data['x1'][i]
        maxd = data['x2'][i]
        max_ = cnt
    if cnt < min_:
        mins = data['x1'][i]
        mind = data['x2'][i]
        min_ = cnt
    calc_cnt.append(cnt)
#calc_cnt = [1, 2, 3, 4]
gn = int((max(calc_cnt) - min(calc_cnt))/10000)
print(max(calc_cnt))
print(maxs)
print(maxd)
print(min(calc_cnt))
print(mins)
print(mind)


plt.hist(calc_cnt, bins = 10)
plt.xlabel("flops")
plt.ylabel("count")
plt.savefig("norm_count.png")
# plt.text()
plt.show()