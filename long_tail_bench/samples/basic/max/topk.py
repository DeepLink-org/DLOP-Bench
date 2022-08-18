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
            


with open("./max.all.json", 'r') as f:
    data = json.load(f)

cnt_dict = {}
print(min(len(data['input']), len(data['dim'])))
for i in range(min(len(data['input']), len(data['dim']))):
    key = []
    if data['input'][i][0] == 99999:
        continue
    for d in data['input'][i]:
        key.append(d)
    key.append(data['dim'][i])
    keyt = tuple(key)
    # print(keyt)
    if keyt in cnt_dict.keys():
        cnt_dict[keyt] += 1
    else:
        cnt_dict[keyt] = 1
d_order=sorted(cnt_dict.items(),key=lambda x:x[1],reverse=True)
print(len(d_order))
d_order = d_order[:10]
print(d_order)
#print#print(d_order[0][0][:-2])

json_dict = {'input':[], 'dim':[]}
for elem in d_order:
    json_dict['input'].append(list(elem[0][:-1]))
    json_dict['dim'].append([elem[0][-1]])
print(json_dict)
with open('max.json', 'w') as json_file:
    json.dump(json_dict, json_file)




f = pd.read_csv("../../../../perf_result/dropout_perf.csv")
# print(f['item_0'][0][0][:-2])
# print(f)
# print(type(f["item_1"].values[0]))
for j in range(len(d_order)): 
    for i in range(len(f["time_cost"].values)): 
    # print(tuple(f['item_0'].values))
        if tuple(string2list(f['item_0'][i])) == d_order[j][0][:-3]:
            print(str(d_order[j][0]) + ": " + str(float(f['item_1'][i][1:-1])) + " : " + str(f["time_cost"].values[i] * 1000) + 'ms')
            break
