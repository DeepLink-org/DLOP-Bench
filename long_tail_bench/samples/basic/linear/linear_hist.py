import json
import matplotlib.pyplot as plt 

with open("linear_all.json", 'r') as f:
    data = json.load(f)
calc_cnt = []
max_ = 0
maxs = []
mins = []
maxd = []
mind = []
min_ = 999999999
for i in range(len(data['input'])):
    cnt = 1
    for d in range(len(data['input'][i])):
        cnt = cnt * data['input'][i][d]
    for d in range(len(data['weight'][i])):
        cnt = cnt * data['weight'][i][d]
    # if len(data['weight'][i]) == 1:
    #     cnt = cnt / data['weight'][i][0]
    # else:
    #     cnt = cnt / data['weight'][i][1]
    cnt = cnt / data['weight'][i][-1]
    if cnt > max_:
        maxs = data['input'][i]
        maxd = data['weight'][i]
        max_ = cnt
    if cnt < min_:
        mins = data['input'][i]
        mind = data['weight'][i]
        min_ = cnt
    calc_cnt.append(cnt)
#calc_cnt = [1, 2, 3, 4]
n, bins, _ = plt.hist(calc_cnt, bins=10, edgecolor='k', density=False)
for i in range(len(n)):
    plt.text((bins[i]+bins[i+1])/2, n[i]*1.01, int(n[i]), color = 'black', fontsize=10, horizontalalignment="center")

plt.xlabel('GFLOPs')
plt.ylabel('Frequency of occurrence')
# plt.yscale("log", base=10)
# plt.text()
# plt.show()
plt.savefig('linear_top10_flops.pdf')

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
plt.savefig("linear_count.png")
# plt.text()
plt.show()