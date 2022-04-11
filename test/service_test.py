#!/usr/local/bin/python3
# -*- coding:utf-8 -*-

import argparse
from pathlib import Path
import requests
import json
import cv2
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random
import warnings
import shutil
from model.config import logo_id_to_name
from multiprocessing import Pool, Manager
warnings.filterwarnings('ignore')
import time

#image_dir = "/data01/xu.fx/dataset/LOGO_DATASET/comb_data/yolo_dataset_comb_364bs_634ks/JPEGImages/twins40perc_images"
image_dir = "/data01/xu.fx/dataset/NEW_RAW_INCREASE_DATA/fordeal_images_2w"
out_pred_img_dir = "/data01/xu.fx/dataset/NEW_RAW_INCREASE_DATA/fordeal_images_2w_pred_0.5"
label_dir = None#"/data01/xu.fx/dataset/LOGO_DATASET/comb_data/yolo_dataset_comb_364bs_634ks/JPEGImages/twins20perc_images_labels"

out_dir = None#"/data01/xu.fx/dataset/LOGO_DATASET/comb_data/yolo_dataset_comb_364bs_634ks/JPEGImages/twins40perc_det_plus"
iou_dir = None#"/data01/xu.fx/dataset/LOGO_DATASET/comb_data/yolo_dataset_comb_364bs_634ks/JPEGImages/twins40perc_comhit_plus"
white_sample_dir = None#"/data01/xu.fx/dataset/LOGO_DATASET/comb_data/yolo_dataset_comb_364bs_634ks/JPEGImages/twins40perc_white_sample_plus"

# image_dir = "/data01/xu.fx/dataset/LOGO_DATASET/comb_data/yolo_dataset_comb_364bs_634ks/JPEGImages/test/images"
#
# out_pred_img_dir = None#"/data01/xu.fx/dataset/LOGO_DATASET/comb_data/yolo_dataset_comb_364bs_634ks/JPEGImages/twins20perc_images_pred"
# label_dir = "/data01/xu.fx/dataset/LOGO_DATASET/comb_data/yolo_dataset_comb_364bs_634ks/JPEGImages/test/labels"
#
# out_dir = "/data01/xu.fx/dataset/LOGO_DATASET/comb_data/yolo_dataset_comb_364bs_634ks/JPEGImages/test_4w_det"
# iou_dir = "/data01/xu.fx/dataset/LOGO_DATASET/comb_data/yolo_dataset_comb_364bs_634ks/JPEGImages/test_4w_comhit"
# white_sample_dir = "/data01/xu.fx/dataset/LOGO_DATASET/comb_data/yolo_dataset_comb_364bs_634ks/JPEGImages/test_4w_white_sample"


WORKERS = 1
iou_thred = 0.5

logo_id_to_name_brand = [i.split("-")[0] for i in logo_id_to_name]
logo_id_to_name_brand = list(set(logo_id_to_name_brand))
base_url = 'http://10.58.14.38:55902'
#base_url = 'https://ai-brand-logo-tmstg.tongdun.cn/'

#10.58.14.38:55902
if out_dir:
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        shutil.rmtree(out_dir)
        os.makedirs(out_dir)
if iou_dir:
    if not os.path.exists(iou_dir):
        os.makedirs(iou_dir)
    else:
        shutil.rmtree(iou_dir)
        os.makedirs(iou_dir)
if white_sample_dir:
    if not os.path.exists(white_sample_dir):
        os.makedirs(white_sample_dir)
    else:
        shutil.rmtree(white_sample_dir)
        os.makedirs(white_sample_dir)

BINARY_API_ENDPOINT = "{}/v2/logo_brand_rec".format(base_url)
image_list = [p for p in Path(image_dir).rglob('*.*')]

find_num = 0
different_num = 0
total_num = len(image_list)
random.shuffle(image_list)
def iou(box1,box2):
    x01,y01,x02,y02=box1
    x11, y11, x12, y12 = box2
    lx = abs((x01+x02)/2-(x11+x12)/2)
    ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
    sax = abs(x01-x02)
    sbx = abs(x11 - x12)
    say = abs(y01 - y02)
    sby = abs(y11 - y12)
    if lx<=(sax+sbx)/2 and ly<=(say+sby)/2:
        bxmin = max(box1[0],box2[0])
        bymin = max(box1[1],box2[1])
        bxmax = min(box1[2],box2[2])
        bymax = min(box1[3],box2[3])
        bwidth = bxmax-bxmin
        bhight = bymax-bymin
        inter = bwidth*bhight
        union = (box1[2]-box1[0])*(box1[3]-box1[1])+(box2[2]-box2[0])*(box2[3]-box2[1])-inter
        return inter/union
    else:
        return 0
def yolotxt_to_voc(h,w,yolobox):
    x1 = yolobox[0] * w - yolobox[2] * w / 2
    y1 = yolobox[1] * h - yolobox[3] * h / 2
    x2 = yolobox[0] * w + yolobox[2] * w / 2
    y2 = yolobox[1] * h + yolobox[3] * h / 2
    return [x1,y1,x2,y2]
def det_server_func(image_list):
    label_index = []
    pred_index = []

    for image_path in tqdm(image_list[:]):
        if image_path.name == ".DS_Store":
            continue
        image_path = str(image_path)
        img = cv2.imread(image_path)
        h, w, _ = img.shape
        file_name = image_path.split('/')[-1]
        payload = {'imageId': '00003'}
        file_temp = [('img', (file_name, open(image_path, 'rb'), 'image/jpeg'))]
        resq1 = requests.request
        try:
            response = resq1("POST", BINARY_API_ENDPOINT, data=payload, files=file_temp)
        except:
            print("request error")
            continue
        result = json.loads(response.text)

        #print(result)
        if 'res' in result:
            pred = result['res']
            box_pred_list = []
            if pred==[]:
                pred_index.append("white sample")
            else:
                #find_num+=1
                logo_list = []
                for logo_instance in pred:
                    logo = logo_instance['logo_name']
                    logo_list.append(logo)
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
                box_pred_num.value += len(logo_list)
                logo_list.sort()
                brand_max_prd = max(logo_list, key=logo_list.count)
                pred_index.append(brand_max_prd)
                if out_pred_img_dir:
                    if not os.path.exists(out_pred_img_dir):
                        os.makedirs(out_pred_img_dir)
                    cv2.imwrite(os.path.join(out_pred_img_dir, pred_index[-1] + "_" + file_name), img)

        #print(pred_index)
        if label_dir:
            postfix = os.path.splitext(file_name)[-1]
            lens = len(postfix)
            imglen = len(file_name)
            label_path = os.path.join(label_dir, file_name[:imglen - lens] + '.txt')
            #print(label_path)
            try:
                txt = np.loadtxt(label_path).reshape(-1, 5)
            except:
                txt = np.array([])
            if txt.shape[0] == 0:
                label_index.append("white sample")
            else:
                res = list(txt[:, 0].astype(int))
                box_label_list = []
                #print(res)
                #print(box_pred_list)
                if iou_dir:
                    hit = 0
                    if len(box_pred_list)>=1:
                        for box in txt[:, 1:]:
                            box_label_list.append(yolotxt_to_voc(h,w,box))
                            #print(h,w,box)
                            #print(box_label_list)
                        for box_gt in box_label_list:
                            for box_pred in box_pred_list:
                                #print(box_pred,box_gt)
                                #print(iou(box_pred,box_gt))
                                iou_ = iou(box_gt,box_pred)
                                if iou_ > iou_thred and iou_<=1:
                                    comhit_num.value+=1
                                    hit = 1
                                    #print("common hit !!!")
                                    cv2.rectangle(img, (int(box_gt[0]), int(box_gt[1])), (int(box_gt[2]), int(box_gt[3])), [0, 255, 0], 2)
                        if hit:
                            #for box_gt in box_label_list:
                            #    cv2.rectangle(img, (int(box_gt[0]), int(box_gt[1])), (int(box_gt[2]), int(box_gt[3])), [0, 255, 0], 2)
                            cv2.imwrite(os.path.join(iou_dir, "CommonHit_" + pred_index[-1] + "_" + file_name), img)
                            shutil.copy(os.path.join(image_dir,file_name),os.path.join(white_sample_dir,file_name))

                res_brand = [logo_id_to_name[min(b,len(logo_id_to_name)-1)].split("-")[0] for b in res]
                res_brand.sort()
                box_gt_num.value+=len(res_brand)
                brand_max_label = max(res_brand, key=res_brand.count)
                label_index.append(brand_max_label)

            if pred_index[-1]!=label_index[-1]:
                if out_dir:
                    save_dir = os.path.join(out_dir,label_index[-1])
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    cv2.imwrite(os.path.join(save_dir, pred_index[-1] + "_" + file_name), img)
                #different_num+=1
                #print("%s is different with gt !!!"%file_name)
    label_index_all.append(label_index)
    pred_index_all.append(pred_index)

label_index_all = Manager().list()
pred_index_all = Manager().list()
#for i in range(WORKERS):
#    label_index_all.append(Manager().list())
#    pred_index_all.append(Manager().list())
comhit_num = Manager().Value("i",0)
box_gt_num = Manager().Value("i",0)
box_pred_num = Manager().Value("i",0)
pool = Pool(processes=WORKERS)
for i in range(0, WORKERS):
    files_ = image_list[i:len(image_list):WORKERS]
    pool.apply_async(det_server_func, (files_,))
pool.close()
pool.join()

#pool.close()
pred_index = []
label_index = []
print(len(pred_index_all))
# print(pred_index_all[1])
print(len(label_index_all))
# print(label_index_all[1])
for i in range(len(label_index_all)):
    label_index+=list(label_index_all[i])
    pred_index+=list(pred_index_all[i])
#print("pred：", pred_index)


if label_dir:
    print("box gt num:", box_gt_num.value)
    print("box pred num:", box_pred_num.value)
    print("comhit num:", comhit_num.value)
    print("total pic num:", total_num)
    print("comhit rate:", comhit_num.value / box_gt_num.value)
    print("-"*100)
    #print("label：", label_index)
    classes = list(set(label_index)).sort()
    matrix_result = confusion_matrix(label_index, pred_index,labels=classes)
    class_result = classification_report(label_index, pred_index,labels=classes)
    #print(class_result)
else:
    print("box pred num:", box_pred_num.value)
    print("total pic num:", total_num)
if out_dir and label_dir:
    with open(os.path.join(out_dir,"class_result.txt"),"w") as t:
        t.write(class_result)
    plt.figure(figsize=(20,20),dpi=300)
    plt.imshow(matrix_result, cmap=plt.cm.Greens)
    indices = range(len(matrix_result))
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    plt.colorbar()
    plt.xlabel('y_pred')
    plt.ylabel('y_true')
    for first_index in range(len(matrix_result)):
        for second_index in range(len(matrix_result[first_index])):
            plt.text(first_index, second_index, matrix_result[first_index][second_index])
    plt.savefig(os.path.join(out_dir,"matrix_result.png"))

#plt.show()




