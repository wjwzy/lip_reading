from LipReading.ResNet import ResNet
from LipReading.opts import args
from LipReading.utils import predict

def test(args):
    num_class = 100
    test_data_path = args.test_data_path
    vocab_path = args.vocab_path
    model_save_path = args.model_save_path
    device = args.device
    eval_batch = args.eval_batch

    model = ResNet(1, num_class)  # 导入网络
    # model = resnet18(1, num_class)
    model.to(device)

    print('预测', '='*100)
    predict(model, eval_batch, model_save_path, test_data_path,
            vocab_path=vocab_path, result_to_save='results.csv', device=device)

if __name__ == '__main__':
    test(args)
