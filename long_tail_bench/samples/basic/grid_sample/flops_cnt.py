import json
import matplotlib.pyplot as plt

with open("./grid_sample.all.json", 'r') as f:
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
for i in range(len(data['input'])):
    input_size = 1
    for d in range(len(data['input'][i])):
        input_size = input_size * data['input'][i][d]
    grid_size = 1
    for d in range(len(data['grid'][i])):
        grid_size = grid_size * data['grid'][i][d]

    cnt = 1
    for d in range(2):
        cnt = cnt * data['input'][i][d]
    for d in range(data['grid'][i][-1]):
        cnt = cnt * data['grid'][i][-2 - d]

    cnt = cnt + input_size + grid_size
    cnt = cnt * 4

    if cnt > max_:
        maxs1 = data['input'][i]
        maxs2 = data['grid'][i]
        max_ = cnt
    if cnt < min_:
        mins1 = data['input'][i]
        mins2 = data['grid'][i]
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
print("min:")
print(min(calc_cnt))
print(mins1)
print(mins2)


calc_cnt = [i/1000000 for i in calc_cnt]

n, bins, _ = plt.hist(calc_cnt, bins=10, edgecolor='k', density=False)
for i in range(len(n)):
    plt.text((bins[i]+bins[i+1])/2, n[i]*1.01, int(n[i]), color = 'black', fontsize=10, horizontalalignment="center")  

plt.xlabel('Mbyte')
plt.ylabel('Frequency of occurrence')
#plt.yscale("log", base=10)

plt.savefig('grid_sample.flops_cnt.jpg')

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
