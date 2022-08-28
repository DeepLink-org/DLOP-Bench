#-*- coding:utf-8 -*-
 
 
import csv
 
 
saveDict={"用户1":"密码1","用户2":"密码2"}
fileName="filename.csv"
##保存文件
with open(fileName,"w") as csv_file:
    writer=csv.writer(csv_file)
    for key,value in saveDict.items():
        writer.writerow([key,value])