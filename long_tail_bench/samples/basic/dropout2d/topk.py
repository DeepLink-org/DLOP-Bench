import json
import pandas as pd
import matplotlib.pyplot as plt 

def string2list(s):
    ret = []
    s_l = len(s)
    start = 1
    for i in range(s_l):
        if s[i] == ',':
            ret.append(int(s[start:i]))
            start = i + 2
    ret.append(int(s[start:s_l-1]))
    return ret

def parentheses_change(s):
    # change '(x, y)' to '[x, y]'
    s_length = len(s)
    ret = ""
    for i in range(s_length):
        if s[i] == '(' and s[i+1] != '[':
            ret += "["
        elif s[i] == ')' and s[i-1] != ']':
            ret += "]"
        else:
            ret += s[i]
    return ret

with open("./dropout2d.all.json", 'r') as f:
    data = json.load(f)

cnt_dict = {}
for i in range(len(data['input_size'])):
    key = []
    for d in data['input_size'][i]:
        key.append(d)
    key.append(data['p'][i][0])
    key.append(data['training'][i][0])
    key.append(data['inplace'][i][0])
    keyt = tuple(key)
    print(keyt)
    if keyt in cnt_dict.keys():
        cnt_dict[keyt] += 1
    else:
        cnt_dict[keyt] = 1
d_order=sorted(cnt_dict.items(),key=lambda x:x[1],reverse=True)
d_order = d_order[:10]
print(d_order)
#print(d_order[0][0][:-2])


json_dict = {'input_size':[], 'p':[], 'training':[], 'inplace':[]}
for elem in d_order:
    json_dict['input_size'].append(list(elem[0][:-3]))
    json_dict['p'].append((elem[0][-3]))
    json_dict['training'].append((elem[0][-2]))
    json_dict['inplace'].append((elem[0][-1]))
print(json_dict)
with open('dropout2d.json', 'w') as json_file:
    json.dump(json_dict, json_file)


f = pd.read_csv("../../../../perf_result/dropout2d_perf.csv")
# print(f)
for j in range(len(d_order)): 
    for i in range(len(f["time_cost"].values)): 
        if tuple(string2list(f['item_0'][i])) == d_order[j][0][:-3]:
            print(str(d_order[j][0]) + ": " + str(float(f['item_1'][i][1:-1])) + " : " + str(f["time_cost"].values[i] * 1000) + 'ms')
