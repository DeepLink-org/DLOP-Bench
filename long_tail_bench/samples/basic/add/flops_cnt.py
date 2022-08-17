import json
import matplotlib.pyplot as plt

with open("./add.all.json", 'r') as f:
    data = json.load(f)
calc_cnt = []
max_ = 0
maxs1 = []
mins1 = []
maxs2 = []
mins2 = []

maxd = []
mind = []
min_ = 9999999
for i in range(len(data['add1'])):
    cnt = 1
    for d in range(len(data['add1'][i])):
        cnt = cnt * data['add1'][i][d]
    if cnt > max_:
        maxs1 = data['add1'][i]
        max_ = cnt
    if cnt < min_:
        mins1 = data['add1'][i]
        min_ = cnt
    calc_cnt.append(cnt)
# calc_sum = sum(calc_cnt)
# print(calc_sum)
# calc_cnt = [i/calc_sum for i in calc_cnt]
# print(calc_cnt)
# gn = int((max(calc_cnt) - min(calc_cnt))/10000)
print("max:")
print(max(calc_cnt))
print(maxs1)
print("min:")
print(min(calc_cnt))
print(mins1)

# plt.hist(calc_cnt, bins=20, edgecolor='k', density=1)
n, bins, _ = plt.hist(calc_cnt, bins=10, edgecolor='k', density=False)
for i in range(len(n)):
    plt.text((bins[i]+bins[i+1])/2, n[i]*1.04, int(n[i]), color = 'black', fontsize=10, horizontalalignment="center")  

plt.xlabel('Output shape size')
plt.ylabel('Count')
plt.yscale("log", base=10)
# plt.text()
# plt.show()
plt.savefig('add.flops_cnt.jpg')

'''
innermost = 0
non_innermost = 0
reduce_ = []
maxr = 0
minr = 999999999
maxrs = []
minrs = []
for i in range(len(data['input_shape'])):
    if data['dim'][i] == -1 or data['dim'][i] == (len(data['input_shape'][i]) - 1):
        innermost += 1
    else:
        non_innermost += 1
    reduce_.append(data['input_shape'][i][data['dim'][i]])
    if maxr < data['input_shape'][i][data['dim'][i]]:
        maxrs = data['input_shape'][i]
        #maxrd = data['dim'][i]
        maxr = data['input_shape'][i][data['dim'][i]]
        maxd = data['dim'][i]
    print(data['input_shape'][i][data['dim'][i]])
    if data['input_shape'][i][data['dim'][i]] < minr:
        minrs = data['input_shape'][i]
        minr = data['input_shape'][i][data['dim'][i]]
        mind = data['dim'][i]
       

print(maxrs)
print(maxd)
print(minrs)
print(mind)
plt.hist(reduce_, bins = 10)
plt.show()
#print(innermost)
#print(non_innermost)

'''
