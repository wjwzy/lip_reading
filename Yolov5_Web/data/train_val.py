import os
import os.path
import random
from shutil import move

def division():
    img_path = "../dataset/images/train" # 图片文件所在的目录
    label_path = "../dataset/labels/train"
    new_path = "../dataset/images/val"
    new_label = "../dataset/labels/val"
    files = os.listdir(img_path)  # 得到文件夹下所有文件名称
    for file in files:  # 遍历文件夹
        if random.random() > 0.9:
            print(file)
            img_data = os.path.join(img_path,file)
            move(img_data,new_path)
            file = file[:-4]
            label_data = os.path.join(label_path,file + '.txt')
            move(label_data,new_label)


if __name__ == '__main__':
    division()