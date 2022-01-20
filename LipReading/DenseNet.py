import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=(1,2,2), stride=(1,2,2)))


class Dense3D(torch.nn.Module):
    def __init__(self, in_c, num_class, growth_rate=32, num_init_features=64, bn_size=4, drop_rate=0):
        super(Dense3D, self).__init__()
        #block_config = (6, 12, 24, 16)
        self.debug_log = True
        block_config = (4, 8, 12, 8)

        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(in_c, num_init_features, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))),
        ]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):

            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)

            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # 最后再一次批次的标准化
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        self.gru1 = nn.GRU(536*3*3,256, bidirectional=True, batch_first=True)
        self.gru2 = nn.GRU(512, 256, bidirectional=True, batch_first=True)


        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_class)
        )

    def forward(self, x, targets=None):
        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()
        f2 = self.features(x)
        if self.debug_log:
            print("shape1:", f2.shape)
        f2 = f2.permute(0, 2, 1, 3, 4).contiguous()
        if self.debug_log:
            print("shape2:", f2.shape)
        B, T, C ,H ,W = f2.size()
        if self.debug_log:
            print("shape3:", f2.shape)
        f2 = f2.view(B, T, -1)
        if self.debug_log:
            print("shape4:", f2.shape)
        f2, _ = self.gru1(f2)
        if self.debug_log:
            print("shape5:", f2.shape)
        f2, _ = self.gru2(f2)
        if self.debug_log:
            print("shape6:", f2.shape)
        f2 = self.fc(f2)
        if self.debug_log:
            print("shape7:", f2.shape)
        logit = f2.log_softmax(-1)
        logit = torch.sum(logit, dim=1)
        result = (logit)

        # loss
        if torch.is_tensor(targets):
            log_sm = torch.mean(-F.log_softmax(f2, -1), dim=1)
            # log_sm = -F.log_softmax(out, -1)
            loss = log_sm.gather(dim=-1, index=targets[:, None]).squeeze()
            result = (logit, loss)

        return result


if (__name__ == '__main__'):
    data = torch.randn(4, 3, 12, 112, 112)
    m = Dense3D(3,100)
    print(m(data).size())


