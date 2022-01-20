import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train_path", default=r'E:/DataSet/new_bank_data/train_data/lip100/', type=str, help='lip_train folder path') # 训练原数据
parser.add_argument("--test_path", default=r'D:/DataSet/new_bank_data/test_data/lip_test/', type=str, help='lip_test folder path') # 无标签的随机数据
parser.add_argument("--label_path", default=r'E:/DataSet/new_bank_data/train_data/lip_train.txt', type=str, help='lip_train.txt file path') # 标签文件
parser.add_argument("--save_path", default='data', type=str, help='the save path of the web_data') # 数据存放的目录
parser.add_argument("--test_data_path", default='data/test_data.dat', type=str,help='test web_data path') # 预处理之后的无标签数据
parser.add_argument("--data_path", default='data/train_data100.dat', type=str,help='train web_data path') # 预处理之后的训练数据
parser.add_argument("--model_save_path", default='./model/model.pt', type=str,help='the path model to save') # 训练的模型
parser.add_argument("--vocab_path", default='data/vocab100.txt', type=str,help='vocab path') # 分类的标签文件
parser.add_argument("--save_img", default='E:/Lip_Code/Web_data/save_img/', type=str, help='the save path of the web_data') # 切帧完成后存放图片的目录
parser.add_argument("--cut_img", default='E:/Lip_Code/Web_data/cut_img/', type=str, help='') # 裁剪完成后存放的图片
parser.add_argument("--save_video", default='E:/Lip_Code/Web_data/uploads/', type=str, help='the save videos') # 前端获取的视频
parser.add_argument("--num_class", default=100, type=int) # 类别数
parser.add_argument("--batch_size", default=4, type=int)
parser.add_argument("--epochs", default=50, type=int)
parser.add_argument("--device", default='cuda:0', type=str)
parser.add_argument("--lr", default=0.0005, type=float) # 学习率
parser.add_argument("--grad_clip", default=0.5, type=float) # 裁剪的梯度
parser.add_argument("--log_step", default=50, type=int, help='print information interval') # 打印日志的时间步数
parser.add_argument("--num_eval", default=0.1, type=int, help='number of verification set') # 测试集样本比例
parser.add_argument("--eval_batch", default=4, type=int, help='batch size of verify') # 测试批量大小
parser.add_argument('--load_cache', action='store_false') # 是否有缓存数据

args = parser.parse_args()
