import json
import pandas as pd
import matplotlib.pyplot as plt 
import os

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

def string2string(s1, s2, s3):
    s1 = s1[1 : len(s1)-1]
    s2 = s2[1 : len(s2)-1]
    s3 = s3[1 : len(s3)-1]
    ret = '(' + s1 + ', ' + s2 + ', ' + s3 + ')'
    return ret

def tuple2string(t):
    return str(t)

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
            


with open("linear_all.json", 'r') as f:
    data = json.load(f)

cnt_dict = {}
for i in range(len(data['input'])):
    key = []
    for d in data['input'][i]:
        key.append(d)
    for d in data['weight'][i]:
        key.append(d)
    for d in data['bias'][i]:
        key.append(d)
    # key.append(data['mat2'][i])
    keyt = tuple(key)
    if keyt in cnt_dict.keys():
        cnt_dict[keyt] += 1
    else:
        cnt_dict[keyt] = 1
d_order=sorted(cnt_dict.items(),key=lambda x:x[1],reverse=True)
d_order = d_order[:10]
print(d_order)
#print(d_order[0][0][:-2])

perf_result_path = os.path.dirname(os.path.dirname((os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
dirpath = os.path.join(perf_result_path, "perf_result", "linear_perf.csv")
f = pd.read_csv(dirpath)
# print(f['item_0'][0][0][:-2])
# print("++++++++++++++++++")
print(f)
# print("------------------")
 
for i in range(len(d_order)):
    for j in range(len(f["time_cost"].values)):
        # if tuple(string2list(f['item_0'][i])) == d_order[j][0][:-3] and string2list(f['item_1'][i]) == list(d_order[j][0][-3:-1]):
        if string2string(f['item_0'][j], f['item_1'][j], f['item_2'][j]) == tuple2string(d_order[i][0]):
            print(str(d_order[i][0]) + ": " + str(f["time_cost"].values[j] * 1000) + 'ms')
