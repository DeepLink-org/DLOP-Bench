import json
import matplotlib.pyplot as plt 

with open("randperm_all.json", 'r') as f:
    data = json.load(f)
calc_cnt = []
max_ = 0
maxs = []
mins = []
maxd = []
mind = []
min_ = 999
for i in range(len(data['n'])):
    cnt = data['n'][i]
    if cnt > max_:
        maxs = data['n'][i]
        maxd = data['n'][i]
        max_ = cnt
    if cnt < min_:
        mins = data['n'][i]
        mind = data['n'][i]
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
plt.savefig("randperm_count.png")
# plt.text()
plt.show()