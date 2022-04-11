#!/usr/local/bin/python3
# -*- coding:utf-8 -*-

import argparse
from pathlib import Path
import requests
import json
import cv2
import os
from tqdm import tqdm
# import numpy as np
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# import random
import warnings
import shutil
#from model.config import logo_id_to_name
#from multiprocessing import Pool, Manager
warnings.filterwarnings('ignore')
#import time
from multiprocessing import Pool, Manager

image_dir = "/data01/xu.fx/dataset/LOGO_DATASET/fordeal_test_data_total/brand_total_raw_data"
out_pred_img_dir = "/data01/xu.fx/dataset/LOGO_DATASET/fordeal_test_data_total/online_comlogo3"
WORKERS = 30
save_label_json = "/data01/xu.fx/dataset/LOGO_DATASET/fordeal_test_data_total/online_comlogo3.json"

if out_pred_img_dir:
    if not os.path.exists(out_pred_img_dir):
        os.makedirs(out_pred_img_dir)
    #else:
        #shutil.rmtree(out_pred_img_dir)
        #os.makedirs(out_pred_img_dir)

base_url = 'http://10.58.14.38:55902'
#base_url = 'http://10.57.31.15:5032'
#base_url = 'https://ai-brand-logo-tmstg.tongdun.cn'

#10.58.14.38:55902


BINARY_API_ENDPOINT = "{}/v2/logo_brand_rec".format(base_url)
image_list = [p for p in Path(image_dir).rglob('*.*')][:]
print(len(image_list))
with open('/data01/xu.fx/comtools/human_label_to_model_label/l2l_dict.json', 'r') as f:
    l2l_data = json.load(f)

find_num = 0
different_num = 0
total_num = len(image_list)
#random.shuffle(image_list)

def det_server_func(image_list,save_json_dict):
    for image_path in tqdm(image_list[:]):
        if image_path.name == ".DS_Store":
            continue
        try:
            image_path = str(image_path)
            img = cv2.imread(image_path)
            h, w, _ = img.shape
            file_name = image_path.split('/')[-1]
            payload = {'imageId': '00003'}
            file_temp = [('img', (file_name, open(image_path, 'rb'), 'image/jpeg'))]
            resq1 = requests.request
            try:
                response = resq1("POST", BINARY_API_ENDPOINT, data=payload, files=file_temp)
            except Exception as e:
                print(e)
                print(file_name)
                continue
            result = json.loads(response.text)
            if 'res' in result:
                pred = result['res']
                #print(pred)
                box_pred_list = []
                logo_list_human = []
                if pred==[]:
                    #pred_index.append("white sample")
                    brand_name = "empty"
                    logo_list_human.append(brand_name)
                else:
                    #find_num+=1
                    logo_list = []
                    for logo_instance in pred:
                        logo = logo_instance['logo_name']
                        logo_list.append(logo)
                        if logo not in l2l_data:
                            logo = logo.lower().replace(" ", "_")
                        else:
                            logo = l2l_data[logo].split("/")[-1]
                        if logo == "new_york_yankees":
                            logo = "mlb"
                        logo_list_human.append(logo)
                        box = logo_instance['box']
                        score = logo_instance['score']
                        x1 = box['x1']
                        y1 = box['y1']
                        x2 = box['x2']
                        y2 = box['y2']
                        box_pred_list.append([x1,y1,x2,y2])
                        cv2.rectangle(img, (x1, y1), (x2, y2), [0, 0, 255], 2)
                        cv2.putText(img, logo, (x1, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 1, cv2.LINE_AA)
                        cv2.putText(img, str(round(score, 3)), (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 1,
                                    cv2.LINE_AA)
                    #box_pred_num.value += len(logo_list)

                    logo_list.sort()
                    brand_max_prd = max(logo_list, key=logo_list.count)
                    brand_name = brand_max_prd.split("-")[0]
                save_json_dict[file_name] = list(set(logo_list_human))
                if out_pred_img_dir:
                    save_dir = os.path.join(out_pred_img_dir,brand_name)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    #print(os.path.join(save_dir, file_name))
                    try:
                        cv2.imwrite(os.path.join(save_dir, file_name), img)
                    except:
                        print("problem file: ",os.path.join(save_dir, file_name))
            else:
                print("error",result,file_name)
        except Exception as e:
            print(e)
            print(image_path)
            continue

save_json_dict = Manager().dict()
pool = Pool(processes=WORKERS)
for i in range(0, WORKERS):
    imgs = image_list[i:len(image_list):WORKERS]
    pool.apply_async(det_server_func, (imgs,save_json_dict,))
pool.close()
pool.join()

# save_json_dict = {}
# det_server_func(image_list,save_json_dict)
# print(len(save_json_dict))
if save_label_json:
    with open(save_label_json, 'w') as f:
        json.dump(dict(save_json_dict), f)
    #print(save_json_dict)
    print("write len：",len(save_json_dict))
    with open(save_label_json, 'r') as f:
        model_result = json.load(f)
    print("read len:",len(model_result))