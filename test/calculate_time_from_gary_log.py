#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
import json
#from tqdm import tqdm
txt_file = "../output/ai-brand-logo-tm/logs/gray.log"
with open(txt_file,"r") as f:
    lines = f.readlines()[100:]
time = 0
tmp_line = lines[10]
tmp_line = tmp_line.split(" - ")[-1]
time_dict = json.loads(tmp_line)["profile_info"]
for key,value in time_dict.items():
    time_dict[key] = 0.0#float(value.split("ms")[0])
for line in lines:
    tmp_dict = json.loads(line.split(" - ")[-1])["profile_info"]
    for key,value in tmp_dict.items():
        time_dict[key] += float(value.split("ms")[0])   
    #time += float(line.split("\"inference\":")[-1].split("ms")[0].split("\"")[-1])
for key,value in time_dict.items():
    time_dict[key] = round(value/len(lines),3)
print(len(lines))
print(time_dict)
