# -*-coding:utf-8-*-
import numpy as np
import random
import torch
import pickle
from multiprocessing.pool import Pool


def padding_batch(array_batch):
    '''
    将一个batch的样本填充至同样帧数
    :param array_batch: 一个batch大小的样本
    :return: 一个batch的训练数据 tensor shape: (batch_size, 3, time_steps, h, w)
    '''
    data = []
    time_steps = [a.shape[0] for a in array_batch]
    max_timestpe = max(time_steps)
    for i, array in enumerate(array_batch):
        if array.shape[0] != max_timestpe:
            t, h, w, c = array.shape
            # 定义填充的0矩阵
            pad_arr = np.zeros((max_timestpe - t, h, w, c), dtype=np.float32)
            # 两个多维矩阵合并
            array_batch[i] = np.vstack((array, pad_arr))
        data.append(array_batch[i])
    data = np.asarray(data, dtype=np.float32)
    # 转置矩阵
    data = data.transpose((0, 4, 1, 2, 3))
    return data

# 加载模型参数与训练参数
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def get_train_data(array_list, label_list, batch_size, test_data=False):
    iter_list = []
    label_data = []
    num_data = len(array_list)
    num_batch = num_data // batch_size if num_data % batch_size == 0 else num_data // batch_size + 1

    batch_range = list(range(num_batch))
    random.shuffle(batch_range)
    # bar = batch_range

    for i in batch_range:
        start = i * batch_size
        end = (i+1) * batch_size if (i+1) * batch_size < num_data else num_data
        # 按照批次分批处理
        iter_list.append(array_list[start:end])
        if test_data:
            label_data.append(label_list[start:end])
        else:
            label_data.append(torch.tensor(label_list[start:end]))

    pool = Pool()
    # 多线程进行单个批次填充
    train_data = pool.map(padding_batch, iter_list)
    pool.close()
    pool.join()
    # 转tensor
    train_data = [torch.tensor(data_item) for data_item in train_data]

    return train_data, label_data

def split_train_eval(array_list, label_list, num_eval):
    train_data = []
    train_label = []
    eval_data = []
    eval_label = []
    # 分割训练集和测试集
    eval_idx = random.sample(range(len(array_list)),num_eval)
    for i in range(len(array_list)):
        if i not in eval_idx:
            train_data.append(array_list[i])
            train_label.append(label_list[i])
        else:
            eval_data.append(array_list[i])
            eval_label.append(label_list[i])
    return train_data, train_label, eval_data, eval_label

def evals(model, eval_data, eval_label, device):
    model.eval()
    # acc = 0
    # count = 0
    pred_label = []
    true_label = []
    with torch.no_grad():
        for step in range(len(eval_data)):
            # 测试数据与标签
            batch_inputs = eval_data[step].to(device)
            batch_labels = eval_label[step].to(device)
            # 得到测试的结果
            logist = model(batch_inputs)
            logist = torch.argmax(logist, dim=-1)

            # 预测标签与真实标签
            pred_label.append(logist)
            true_label.append(batch_labels)
            # count += logist.size(0)
            # 进行预测标签与真实标签的对比得到正确的数量
            # acc += torch.sum(torch.eq(torch.argmax(logist, dim=-1), batch_labels)).item()
        acc = torch.mean(torch.eq(torch.cat(pred_label), torch.cat(true_label)).float()).item()
    model.train()
    return acc


def predict(model, batch_size, model_path, data_path, vocab_path, result_to_save, device):
    ##############################
    #         模型加载
    ##############################
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device))
    model.eval()
    model.to(device)
    print('加载模型')

    id2label = []
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for word in f:
            id2label.append(word.split(',')[0])

    ##############################
    #         数据加载
    ##############################
    with open(data_path, 'rb') as f:
        test_data = pickle.load(f)
        test_ids = pickle.load(f)
    print('数据加载完成, data num = {}, label num = {}'.format(len(test_data), len(test_ids)))

    # 按照batch分割数据
    test_data, test_ids = get_train_data(test_data, test_ids, batch_size, test_data=True)
    ##############################
    #            预测
    ##############################
    print('预测中...')
    pre_result = []
    with torch.no_grad():
        for step in range(len(test_data)):
            batch_inputs = test_data[step].to(device)
            logist = model(batch_inputs)
            pred = torch.argmax(logist, dim=-1).tolist()
            assert len(pred) == len(test_ids[step])
            for i, ids in enumerate(test_ids[step]):
                pre_result.append(ids + ',' + id2label[pred[i]])
    with open(result_to_save, 'w', encoding='utf-8') as f:
        for line in pre_result:
            f.write(line + '\n')
    print('预测结果已保存至:', result_to_save)

