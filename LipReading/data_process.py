# -*-coding:utf-8-*-
import numpy as np
import cv2
import os
import random
import pickle
from tqdm import tqdm
from imgaug import augmenters as iaa
from LipReading.opts import args

# 对图像进行缩放填充至fixed_side大小
def img_clip(img, fixed_side = 112):
    h, w = img.shape[0], img.shape[1]
    scale = max(w, h) / float(fixed_side)  # 获取缩放比例
    new_w, new_h = int(w / scale), int(h / scale)
    resize_img = cv2.resize(img, (new_w, new_h))  # 按比例缩放

    # 计算需要填充的像素长度
    if new_w % 2 != 0 and new_h % 2 == 0:
        top, bottom, left, right = (fixed_side - new_h) // 2, (fixed_side - new_h) // 2, (fixed_side - new_w) // 2 + 1, (fixed_side - new_w) // 2
    elif new_w % 2 == 0 and new_h % 2 != 0:
        top, bottom, left, right = (fixed_side - new_h) // 2 + 1, (fixed_side - new_h) // 2, (fixed_side - new_w) // 2, (fixed_side - new_w) // 2
    elif new_w % 2 == 0 and new_h % 2 == 0:
        top, bottom, left, right = (fixed_side - new_h) // 2, (fixed_side - new_h) // 2, (fixed_side - new_w) // 2, (fixed_side - new_w) // 2
    else:
        top, bottom, left, right = (fixed_side - new_h) // 2 + 1, (fixed_side - new_h) // 2, (fixed_side - new_w) // 2 + 1, (fixed_side - new_w) // 2

    # 填充图像
    img = cv2.copyMakeBorder(resize_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img

def get_data_info(data_dir):
    data_dirs = os.listdir(data_dir)  #获取文件夹列表
    data_info = {}
    for d in data_dirs:
        num = len(os.listdir(os.path.join(data_dir, d)))
        if num not in data_info:
            data_info[num] = [d]
        else:
            data_info[num].append(d)
    return data_info

def logging(s, log_path='E:/Lip_Code/LipRead2021/LipReading/data/logs/error_log.txt'):
    with open(log_path, 'a+', encoding='utf-8') as f_log:
        f_log.write(s + '\n')

def _sample(cut_img_list, bilater = True):
    data = []
    for img in cut_img_list:
        img = img_clip(img)  # 缩放并填充至112大小
        if bilater and random.random() < 0.6:
            # 引入双边滤波去噪
            img = cv2.bilateralFilter(src=img, d=0, sigmaColor=random.randint(15, 30), sigmaSpace=15)

        # 归一化，转换数据类型 并限定上下界限的大小必须为fixed_side
        img = img.astype(np.float32)
        # 标准化处理
        img -= np.mean(img)  # 减去均值
        img /= np.std(img)  # 除以标准差
        data.append(img)
    return np.array(data)

def get_sample(sample_path, bilater = True):
    img_paths = os.listdir(sample_path)
    img_paths = [int(i.split('.')[0]) for i in img_paths if i.split('.')[0].isdigit()] # 路径切割
    img_paths = ['{}.png'.format(i) for i in sorted(img_paths)] # 对图片按照名称进行排序

    data = []
    for img_name in img_paths:
        img_path = os.path.join(sample_path, img_name)  # 全部拼接并返回路径
        img = cv2.imread(img_path)
        if img is None:  # 对读取的图片进行判空，以免错误转义
            logging('no this picture:{}'.format(img_path))
            print('no this picture!', img_path)
            return None

        img = img_clip(img)  # 缩放并填充至112大小
        if bilater and random.random() < 0.6:
            # 引入双边滤波去噪
            img = cv2.bilateralFilter(src=img, d=0, sigmaColor=random.randint(15, 30), sigmaSpace=15)

        # 归一化，转换数据类型 并限定上下界限的大小必须为fixed_side
        img = img.astype(np.float32)
        # 标准化处理
        img -= np.mean(img)  # 减去均值
        img /= np.std(img)  # 除以标准差
        data.append(img)
    return np.array(data)

def set_sample(sample_path, bilater = True):
    img_paths = os.listdir(sample_path)
    img_paths = [int(i.split('.')[0]) for i in img_paths if i.split('.')[0].isdigit()] # 路径切割
    img_paths = ['{}.png'.format(i) for i in sorted(img_paths)] # 对图片按照名称进行排序

    data = []
    for img_name in img_paths:
        img_path = os.path.join(sample_path, img_name)  # 全部拼接并返回路径
        img = cv2.imread(img_path)
        if img is None:  # 对读取的图片进行判空，以免错误转义
            logging('no this picture:{}'.format(img_path))
            print('no this picture!', img_path)
            return None

        img = img_clip(img)  # 缩放并填充至112大小
        seq = iaa.Sequential([
            iaa.Flipud(1) # 镜像翻转
        ])
        img = seq.augment_images(img)

        if bilater and random.random() < 0.6:
            # 引入双边滤波去噪
            img = cv2.bilateralFilter(src=img, d=0, sigmaColor=random.randint(15, 30), sigmaSpace=15)

        # 归一化，转换数据类型 并限定上下界限的大小必须为fixed_side
        img = img.astype(np.float32)
        # 标准化处理
        img -= np.mean(img)  # 减去均值
        img /= np.std(img)  # 除以标准差
        data.append(img)
    return np.array(data)

def read_data(root_dir, data_info, id2word=None, word2label=None, test_data=False):
    data = []
    labels = []
    sorted_keys = sorted(data_info.keys(), reverse=True)
    for key in sorted_keys:
        # key: num of time step
        if key < 1:
            continue
        for s in tqdm(data_info[key]):
            # s: 一个样本的文件名
            sample = get_sample(os.path.join(root_dir, s))
            if sample is not None:
                data.append(sample)
                if test_data:
                    labels.append(s)
                else:
                    labels.append(word2label[id2word[s]])

            # 增强一倍数据样本
            if test_data == False:
                sample = set_sample(os.path.join(root_dir, s))
                if sample is not None:
                    data.append(sample)
                    labels.append(word2label[id2word[s]])
    return data, labels

def get_vocab(label_file):
    '''
    建立词表
    :param label_file: lip_train.txt文件的位置
    :return: 样本id与词语的对应id2word, 词语与下标的对应word2label
    '''
    id2word = {}
    word2label = {}
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            ids, word = line.strip().split('\t')
            id2word[ids] = word
            if word not in word2label:
                word2label[word] = len(word2label)
        # print(word2label)
    return id2word, word2label
# 保存词表
def save_vocab(word2label, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for word in word2label.keys():
            f.write('{},{}\n'.format(word, word2label[word]))


if __name__ == '__main__':
    # 读取文件与保存文件
    train_path = args.train_path
    label_path = args.label_path
    save_path = args.save_path

    print("数据加载中...")
    # 将所有唇语汉字词语按照类别拼接到txt文件中并保存下来
    id2word, word2label = get_vocab(label_path)
    save_vocab_path = os.path.join(save_path, 'vocab.txt')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        # 解析并保存词表
        save_vocab(word2label, save_vocab_path)

    # 解析并保存训练数据
    data_info = get_data_info(train_path)
    data, labels = read_data(train_path, data_info, id2word, word2label)
    with open(os.path.join(save_path, 'train_data_100.dat'), 'wb') as f:
        pickle.dump(data, f)
        pickle.dump(labels, f)
        f.close()
    print('训练数据已保存')
