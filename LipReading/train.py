import torch
import random
import pickle
from LipReading.ResNet import ResNet
from LipReading.opts import args
from LipReading.utils import get_parameter_number, split_train_eval, get_train_data, evals


def train(args):
    num_class = args.num_class
    save_model = True
    data_path = args.data_path
    model_save_path = args.model_save_path
    batch_size = args.batch_size
    epochs = args.epochs
    device = args.device
    lr = args.lr
    log_step = args.log_step
    grad_clip = args.grad_clip
    num_eval = args.num_eval
    load_cache = args.load_cache

    ##############################
    #         模型加载
    ##############################
    model = ResNet(3,num_class) # 导入网络
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 用Adam进行优化
    print('加载模型')
    num_param = get_parameter_number(model)
    print('total parameter: {}, trainable parameter: {}'.format(num_param['Total'], num_param['Trainable']))

    # 数据加载
    if load_cache:
        # 读取缓存数据
        with open('cache.dat', 'rb') as f:
            train_data = pickle.load(f)
            label_data = pickle.load(f)
            eval_data = pickle.load(f)
            eval_label = pickle.load(f)
        print('加载缓存数据, train data num = {}, eval data num = {}'.format(len(train_data), len(eval_data)))
    else:
        with open(data_path, 'rb') as f:
            train_data = pickle.load(f)
            label_data = pickle.load(f)

        # 数据分割处理
        num_eval = round(len(train_data) * num_eval)
        train_data, label_data, eval_data, eval_label = split_train_eval(train_data, label_data, num_eval)
        print('数据分割完成, train data num = {}, eval data num = {}'.format(len(train_data), len(eval_data)))
        # 分批并填充
        train_data, label_data = get_train_data(train_data, label_data, batch_size)
        eval_data, eval_label = get_train_data(eval_data, eval_label, batch_size)
        # 保存缓存数据
        with open('cache.dat', 'wb') as f:
            pickle.dump(train_data, f)
            pickle.dump(label_data, f)
            pickle.dump(eval_data, f)
            pickle.dump(eval_label, f)
        print('缓存数据完成')
    print('加载完成, train batch num = {}, eval batch num = {}'.format(len(train_data), len(eval_data)))

    # 训练
    best_acc = -1
    pred_label = []
    true_label = []
    # 不断更新迭代
    for epoch in range(1, epochs+1):
        avg_loss = 0
        data_indexs = list(range(len(train_data)))
        random.shuffle(data_indexs)
        for step, data_idx in enumerate(data_indexs):
            batch_inputs = train_data[data_idx].to(device)
            batch_labels = label_data[data_idx].to(device)

            # 数据经过网络得到类别概率与损失
            logist, loss = model(batch_inputs, batch_labels)
            logist = torch.argmax(logist, dim=-1)
            loss = loss.mean()

            # 预测标签与真实标签
            pred_label.append(logist)
            true_label.append(batch_labels)
            avg_loss += loss.item()

            # 打印每log_step的的训练日志
            if step % log_step == 0:
                pred_acc = torch.mean(torch.eq(torch.cat(pred_label), torch.cat(true_label)).float()).item()
                print('epoch={}, step={}, timestep={}, loss={:.3f}, predict acc={:.3f}'.format(
                    epoch, step, batch_inputs.size(2), avg_loss/(step+1), pred_acc))
                pred_label = []
                true_label = []

            optimizer.zero_grad()  # 梯度归零
            loss.backward()  # 反向传播更新梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip) # 梯度裁剪
            optimizer.step()  # 梯度下降更新参数
            step += 1
        # 测试
        acc = evals(model, eval_data, eval_label, device=device)
        print('='*100)
        print('epoch = {}, Avg train loss = {}, Acc = {}'.format(epoch, avg_loss/len(train_data), acc))

        # 判断是否保存模型，保存正确率最高模型
        if save_model and acc >= best_acc:
            model_to_save = model.module if hasattr(model, 'module') else model
            with open(model_save_path, 'wb') as f:
                torch.save(model_to_save.state_dict(), f)
            print('保存模型:', model_save_path)
            best_acc = acc
        print('=' * 100)
    print('训练完成: best acc =', best_acc)


if __name__ == '__main__':
    train(args)
