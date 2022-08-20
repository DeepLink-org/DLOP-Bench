from cmath import nan
import json
import pandas as pd
import matplotlib.pyplot as plt 
import os
import numpy

def string2list(s):
    ret = []
    s_l = len(s)
    if s_l == 2:
        return ret
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
            


with open("normalize_all.json", 'r') as f:
    data = json.load(f)


cnt_dict = {}
for i in range(len(data['input'])):
    key = []
    for d in data['input'][i]:
        key.append(d)
    
    if data['p'][i] == "":
        key.append(2)
    else:
        key.append(data['p'][i])
    
    if data['dim'][i] == "":
        key.append(1)
    else:
        key.append(data['dim'][i])

    if data['eps'][i] == "":
        key.append(1e-12)
    else:
        key.append(data['eps'][i])

    keyt = tuple(key)
    if keyt in cnt_dict.keys():
        cnt_dict[keyt] += 1
    else:
        cnt_dict[keyt] = 1
d_order=sorted(cnt_dict.items(),key=lambda x:x[1],reverse=True)
d_order = d_order[:10]
print(d_order)

topk_dict = {"input": [], "p":[], "dim": [], "eps": [], "out":[]}
for item in d_order:
    topk_dict["input"].append(list(item[0][:-3]))
    topk_dict["p"].append(item[0][-3])
    topk_dict["dim"].append(item[0][-2])
    topk_dict["eps"].append(item[0][-1])
    topk_dict["eps"].append(item[0][-1])
    topk_dict["out"].append(None)
with open('normalize.json', 'w') as json_file:
    json.dump(topk_dict, json_file)

perf_result_path = os.path.dirname(os.path.dirname((os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
dirpath = os.path.join(perf_result_path, "perf_result", "normalize_perf.csv")
f = pd.read_csv(dirpath)
print(f)

print("""\\begin{table}[H]
\\label{tbl:linalg_normalize_top10}
    \\centering
    \\caption{top10 configurations and call times}
\\begin{tabular}{c|c|c}
\\hline""")
print("id & Config ([input], [p=2.0], [dim=1], [eps=1e-12]) & call times \\\\ \hline")
for i in range(len(d_order)):
    print(str(i + 1) + " & " + "(" + str(list(d_order[i][0][:-3])) + ", " + str(list(d_order[i][0][-3:-2])) + ", " + str(list(d_order[i][0][-2:-1])) + ", " + str(list(d_order[i][0][-1:])) +")" + 
    " & " + str(d_order[i][1]) + " \\\\ \hline")
    # for j in range(len(f["time_cost"].values)):
    #     item_0 = string2list(f['item_0'][j])
    #     item_1 = 2.0 if f['item_1'][j] == nan else f['item_1'][j]
    #     item_2 = 1 if f['item_2'][j] == nan else f['item_2'][j]
    #     item_3 = 1e-12 if f['item_3'][j] == nan else f['item_3'][j]
    #     if item_0 == list(d_order[i][0][:-3]) and item_1 == d_order[i][0][-3] and item_2 ==d_order[i][0][-2] and item_3 == d_order[i][0][-1]:
    #         print(str(i + 1) + " & " + "(" + f['item_0'][j] + ", [" + str(item_1) + "], [" + str(item_2) + "], [" + str(item_2) + "])" + " & " + str(d_order[i][1]) + " \\\\ \hline")
            # print(str(d_order[i][0]) + ": " + str(f["time_cost"].values[j] * 1000) + 'ms')
            # break
print("""\\end{tabular}
\\end{table}""")