import json
import matplotlib.pyplot as plt 

with open("./long_tail_bench/samples/basic/softmax/softmax.all.json", 'r') as f:
    data = json.load(f)
calc_cnt = []
max_ = 0
maxs = []
mins = []
maxd = []
mind = []
min_ = 999
for i in range(len(data['input_shape'])):
    cnt = 1
    for d in range(len(data['input_shape'][i])):
        cnt = cnt * data['input_shape'][i][d]
    cnt = cnt * 2 + data['input_shape'][i][data['dim'][i]]
    if cnt > max_:
        maxs = data['input_shape'][i]
        maxd = data['dim'][i]
        max_ = cnt
    if cnt < min_:
        mins = data['input_shape'][i]
        mind = data['dim'][i]
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


#plt.hist(calc_cnt, bins = 10)
#plt.text()
#plt.show()

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


