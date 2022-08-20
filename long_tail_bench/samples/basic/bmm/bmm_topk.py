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

def string2string(s1, s2):
    s1 = s1[1 : len(s1)-1]
    s2 = s2[1 : len(s2)-1]
    ret = '(' + s1 + ', ' + s2 + ')'
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
            


with open("bmm_all.json", 'r') as f:
    data = json.load(f)

cnt_dict = {}
for i in range(len(data['input'])):
    key = []
    for d in data['input'][i]:
        key.append(d)
    for d in data['mat2'][i]:
        key.append(d)
    keyt = tuple(key)
    if keyt in cnt_dict.keys():
        cnt_dict[keyt] += 1
    else:
        cnt_dict[keyt] = 1
d_order=sorted(cnt_dict.items(),key=lambda x:x[1],reverse=True)
d_order = d_order[:10]
print(d_order)

topk_dict = {"input": [], "mat2": []}
for item in d_order:
    topk_dict["input"].append(list(item[0][:-3]))
    topk_dict["mat2"].append(list(item[0][-3:]))
with open('bmm.json', 'w') as json_file:
    json.dump(topk_dict, json_file)


perf_result_path = os.path.dirname(os.path.dirname((os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
dirpath = os.path.join(perf_result_path, "perf_result", "bmm_perf.csv")
f = pd.read_csv(dirpath)
print(f)

print("""\\begin{table}[H]
\\label{tbl:blas_bmm_top10}
    \\centering
    \\caption{top10 configurations and call times}
\\begin{tabular}{c|c|c}
\\hline""")
print("id & Config ([input], [mat2]) & call times \\\\ \hline")
for i in range(len(d_order)):
    for j in range(len(f["time_cost"].values)):
        if string2string(f['item_0'][j], f['item_1'][j]) == tuple2string(d_order[i][0]):
            print(str(i + 1) + " & " + "(" + f['item_0'][j] + ", " + f['item_1'][j] + ")" + " & " + str(d_order[i][1]) + " \\\\ \hline")
            break
            # print(str(d_order[i][0]) + ": " + str(f["time_cost"].values[j] * 1000) + 'ms')
print("""\\end{tabular}
\\end{table}""")