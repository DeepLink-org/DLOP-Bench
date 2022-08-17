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
plt.savefig("ger_count.png")
# plt.text()
plt.show()