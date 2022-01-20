from shutil import copy
import os
from os.path import join as pjoin

# 目录配置
source = 'E:/DataSet/new_bank_data/train_data/lip_train'
new_source = 'E:/DataSet/new_bank_data/train_data/lip100'
data_label = 'web_data/vocab100.txt'
data_root = 'E:/DataSet/new_bank_data/train_data/lip_train.txt'


with open(data_label, 'r', encoding='utf-8') as f:
    # 读取整个文件，并将每一行放到列表中
    lines = f.readlines()
    labels = [line.strip().split(',') for line in lines]

with open(data_root, 'r', encoding='utf-8') as f:
    # 读取整个文件，并将每一行放到列表中
    lines0 = f.readlines()
    labels0 = [line.strip().split('\t') for line in lines0]

for i in range(len(labels)):
    for j in range(len(labels0)):
        if labels[i][0] == labels0[j][1]:
            path_from = os.path.join(source,labels0[j][0])
            path_come = os.path.join(new_source,labels0[j][0])
            os.mkdir(path_come)
            for k in os.listdir(path_from):
                data_png = pjoin(path_from, k)
                copy(data_png,path_come)
    print(i)
