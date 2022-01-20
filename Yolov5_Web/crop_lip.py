import cv2
import os
from os.path import join as pjoin

# 目录配置
source = 'E:/DataSet/new_bank_data/train_data/lip_train'
data_root = 'E:/DataSet/new_bank_data/train_root'

def logging(s, log_path='runs/error/log.txt'):
    with open(log_path, 'a+', encoding='utf-8') as f_log:
        f_log.write(s + '\n')

for i in os.listdir(source):
    source0 = pjoin(source, i)  # 得到子文件夹路径
    data_root0 = pjoin(data_root, i)

    for j in os.listdir(source0):
        source1 = pjoin(source0, j)  # 得到图片路径
        j = os.path.splitext(j)
        ext = j[0] + '.txt'
        data_root1 = pjoin(data_root0, ext)

        if os.path.exists(data_root1) == False:
            logging('{}'.format(data_root1))
            print("无标签:" + data_root1)
            continue  # 对txt判空

        with open(data_root1, 'r', encoding='utf-8') as f:
            # 读取整个文件，并将每一行放到列表中
            lines = f.readlines()
            labels = [line.strip().split(' ') for line in lines]
            labels = labels[0]
            image = cv2.imread(source1)  # 原图路径
            cropped = image[int(labels[2]): int(labels[4]), int(labels[1]):int(labels[3])]
            cv2.imwrite(source1, cropped)  # 保存图片到新的路径

    print(i)
