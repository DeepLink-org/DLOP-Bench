import json
import matplotlib.pyplot as plt 

with open("topk_all.json", 'r') as f:
    data = json.load(f)
calc_cnt = []
max_ = 0
min_ = 999
for i in range(len(data['topk_0'])):
    cnt = 1
    for item in data['topk_0'][i]:
        cnt = cnt * item
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
plt.xlabel("shape size")
plt.ylabel("count")
plt.savefig("topk_count.png")
# plt.text()
plt.show()