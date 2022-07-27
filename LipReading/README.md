
# 使用方法
### 1.处理数据
```
python data_process
```
程序会读取并处理训练集和预测集数据，并在`data/`目录下缓存处理好的训练集文件`train_data.dat`、预测集文件`test_data.dat`以及词表`vocab.txt`

### 2.训练与测试
```
python train.py
```
程序会读取上一步处理的数据集和训练集文件，并根据`batch_size`填充数据，输入模型进行训练。

### 3.预测
```
python predict.py
```
完成训练后，再通过predict.py进行预测（前面同时也预处理好预测数据），并将预测结果保存为`results.csv`

## trick：
图片中提到的线性分类层中的自适应词语边界，实际上就是不将GRU最后输出的隐藏层向量直接连接分类层，
而是将GRU的每个time step的输出连接全连接层，在做sotfmax之后在time step维度相加，
使得每个time step的输出都能为最后的分类做出贡献，原因就在于词语的边界位未知，不一定最后一帧图片刚好表示词语说完。
