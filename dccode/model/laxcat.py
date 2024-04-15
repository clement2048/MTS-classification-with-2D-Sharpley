import torch
import torch.nn as nn
import torch.nn.functional as F
# import random
# import copy
# import math
# import numpy as np
# import matplotlib.pyplot as plt


# 密集插值函数
# 使序列长度增加至指定的长度M
def dense_interpolation(x, M):
    # x: B * C * L
    # x 是一个输入三维张量
    # B:批量大小,batch_size
    # C:通道数，channel_num
    # L:原始信号长度, 测量时间长度吧
    u = [0 for i in range(M)]
    for t in range(1, 1 + x.size(2)):
        s = M * t / x.size(2)
        for m in range(1, M + 1):
            w = (1 - abs(s - m) / M) ** 2
            # unsqueeze用于升维，这里是添加了第三维
            u[m - 1] += w * (x[:, :, t - 1].unsqueeze(-1))
    # cat函数用于拼接张量，按维度-1行进行拼接
    return torch.cat(u, -1)



# 相比与论文中的代码，这里貌似缺少了特征提取
# 输入注意力层
# 继承与nn.Module
class input_attention_layer(nn.Module):
    # p:输入数据的通道数，
    # j:隐藏单元的维度
    def __init__(self, p, j):
        super(input_attention_layer, self).__init__()
        self.weight1 = nn.Parameter(torch.ones(p, 1), requires_grad=True)
        self.bias1 = nn.Parameter(torch.zeros(j), requires_grad=True)
        self.weight2 = nn.Parameter(torch.ones(j, p), requires_grad=True)
        self.bias2 = nn.Parameter(torch.zeros(p), requires_grad=True)

    def forward(self, x):
        # x: B * p * j * l
        l = x.size(3)
        h = [0 for i in range(l)]
        x = x.transpose(1, 3)
        # x: B * l * j * p
        for i in range(l):
            # 论文中的激活函数
            # matmul函数就是矩阵相乘
            # F 是 torch.nn.functional 模块的别名
            tmp = F.relu(torch.matmul(x[:, i, :, :], self.weight1).squeeze(-1) + self.bias1)
            tmp = F.relu(torch.matmul(tmp, self.weight2) + self.bias2)
            # tmp: B * p
            # attn为注意力得分，具体还是没有看明白，但是最终返回了一个p维张量
            # 为每个变量的注意力得分
            attn = F.softmax(tmp, -1).unsqueeze(1)
            h[i] = torch.sum(attn * x[:, i, :, :], -1)
            h[i] = h[i].unsqueeze(-1)  # unsqueeze for cat
            # B * j
        return torch.cat(h, -1)
        # B * j * l


# 时序注意力层
class temporal_attention_layer(nn.Module):
    def __init__(self, j, l):
        super(temporal_attention_layer, self).__init__()
        # 设置参数
        # 这些参数的维度都是论文上的
        self.weight1 = nn.Parameter(torch.ones(l, 1), requires_grad=True)
        self.bias1 = nn.Parameter(torch.zeros(j), requires_grad=True)
        self.weight2 = nn.Parameter(torch.ones(j, l), requires_grad=True)
        self.bias2 = nn.Parameter(torch.zeros(l), requires_grad=True)

    # 前向传播
    def forward(self, x):
        # x: B * j * l
        tmp = F.relu(torch.matmul(x, self.weight1).squeeze(-1) + self.bias1)
        tmp = F.relu(torch.matmul(tmp, self.weight2) + self.bias2)
        attn = F.softmax(tmp, -1).unsqueeze(1)
        # attn: B * 1 * l
        x = torch.sum(attn * x, -1)
        return x


# 分类层
class LaxCat(nn.Module):
    def __init__(self, input_size, input_channel, num_label, hidden_dim=32, kernel_size=64, stride=16):
        super(LaxCat, self).__init__()
        l = int((input_size - kernel_size) / stride) + 1
        # 进行一维卷积
        self.Conv1 = nn.ModuleList([
            nn.Conv1d(1, hidden_dim, kernel_size=kernel_size, stride=stride) for _ in range(input_channel)])
        # 变量注意力评分
        self.variable_attn = input_attention_layer(p=input_channel, j=hidden_dim)
        # 时序注意力评分
        self.temporal_attn = temporal_attention_layer(j=hidden_dim, l=l)
        self.fc = nn.Linear(hidden_dim, num_label)

    # 前向传播函数
    def forward(self, x):
        x = x.transpose(1, 2)
        x = list(x.split(1, 1))
        for i in range(len(x)):
            x[i] = self.Conv1[i](x[i]).unsqueeze(-1)

        x = torch.cat(x, -1).permute(0, 3, 1, 2)
        # x = F.relu(self.Conv1(x)).reshape(B, C, -1, x.size(0))
        x = self.variable_attn(x)
        x = self.temporal_attn(x)
        return self.fc(x)


def main():
    # 创建模型并且转移到GPU上
    stft_m = LaxCat(input_size=256, input_channel=6, num_label=4).cuda()
    # 统计模型参数总数量
    total_params = sum(p.numel() for p in stft_m.parameters())
    print(f'{total_params:,} total parameters.')
    # 输入张量x传递给模型‘stft_m’,并且前向传播，将输出结果
    # train_STFT_model(stft_m, window_size = 64, K = 16)
    x = torch.zeros(3, 256, 6).cuda()
    # 张量x传递给模型进行前向传播，打印输出
    output = stft_m(x)
    print(output.size())


if __name__ == '__main__':
    main()
