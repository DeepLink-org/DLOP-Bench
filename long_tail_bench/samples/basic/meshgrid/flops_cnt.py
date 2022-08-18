import json
import matplotlib.pyplot as plt

with open("./meshgrid.all.json", 'r') as f:
    data = json.load(f)
calc_cnt = []
max_ = 0
maxs1 = []
mins1 = []
maxs2 = []
mins2 = []
maxs3 = []
mins3 = []

maxd = []
mind = []
min_ = 9999999
for i in range(len(data['input0'])):
    cnt = 1
    for d in range(len(data['input0'][i])):
        cnt = cnt * data['input0'][i][d]
    for d in range(len(data['input1'][i])):
        cnt = cnt * data['input1'][i][d]
    for d in range(len(data['input2'][i])):
        cnt = cnt * data['input2'][i][d]
    if cnt > max_:
        maxs1 = data['input0'][i]
        maxs2 = data['input1'][i]
        maxs3 = data['input2'][i]
        max_ = cnt
    if cnt < min_:
        mins1 = data['input0'][i]
        mins2 = data['input1'][i]
        mins3 = data['input2'][i]
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
print(maxs2)
print(maxs3)
print("min:")
print(min(calc_cnt))
print(mins1)
print(mins2)
print(mins3)

# plt.hist(calc_cnt, bins=20, edgecolor='k', density=1)
plt.hist(calc_cnt, bins=20, edgecolor='k', density=False)
plt.xlabel('Output shape size')
plt.ylabel('Count')
# plt.text()
# plt.show()
plt.savefig('meshgrid.flops_cnt.jpg')

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