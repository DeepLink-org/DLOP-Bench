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
            


with open("mv_all.json", 'r') as f:
    data = json.load(f)

cnt_dict = {}
for i in range(len(data['x1'])):
    key = []
    for d in data['x1'][i]:
        key.append(d)
    for d in data['x2'][i]:
        key.append(d)
    keyt = tuple(key)
    if keyt in cnt_dict.keys():
        cnt_dict[keyt] += 1
    else:
        cnt_dict[keyt] = 1
d_order=sorted(cnt_dict.items(),key=lambda x:x[1],reverse=True)
d_order = d_order[:10]
print(d_order)

topk_dict = {"x1": [], "x2": []}
for item in d_order:
    topk_dict["x1"].append(list(item[0][:-1]))
    topk_dict["x2"].append(list(item[0][-1:]))
with open('mv.json', 'w') as json_file:
    json.dump(topk_dict, json_file)

perf_result_path = os.path.dirname(os.path.dirname((os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
dirpath = os.path.join(perf_result_path, "perf_result", "mv_perf.csv")
f = pd.read_csv(dirpath)
print(f)

print("""\\begin{table}[H]
\\label{tbl:blas_mv_top10}
    \\centering
    \\caption{top10 configurations and call times}
\\begin{tabular}{c|c|c}
\\hline""")
print("id & Config ([input], [vec]) & call times \\\\ \hline")
for i in range(len(d_order)):
    for j in range(len(f["time_cost"].values)):
        if tuple(string2list(f['item_0'][j])) == d_order[i][0][:2] and string2list(f['item_1'][j]) == list(d_order[i][0][2:]):
            print(str(i + 1) + " & " + "(" + f['item_0'][j] + ", " + f['item_1'][j] + ")" + " & " + str(d_order[i][1]) + " \\\\ \hline")
            # print(str(d_order[i][0]) + ": " + str(f["time_cost"].values[j] * 1000) + 'ms')
            break
print("""\\end{tabular}
\\end{table}""")