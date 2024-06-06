import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
# import random
# import copy
# import math
# import numpy as np
# import matplotlib.pyplot as plt


'''
# # 密集插值函数
# # 使序列长度增加至指定的长度M
# def dense_interpolation(x, M):
#     # x: B * C * L
#     # x 是一个输入三维张量
#     # B:批量大小,batch_size
#     # C:通道数，channel_num
#     # L:原始信号长度, 测量时间长度吧
#     u = [0 for i in range(M)]
#     for t in range(1, 1 + x.size(2)):
#         s = M * t / x.size(2)
#         for m in range(1, M + 1):
#             w = (1 - abs(s - m) / M) ** 2
#             # unsqueeze用于升维，这里是添加了第三维
#             u[m - 1] += w * (x[:, :, t - 1].unsqueeze(-1))
#     # cat函数用于拼接张量，按维度-1行进行拼接
#     return torch.cat(u, -1)
'''




'''
输入注意力层
p:特征组合数
j:隐藏单元的维度
'''
# 继承于nn.Module
class input_attention_layer(nn.Module):
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


'''
分类层
time_num:采样时间点的个数
feature_num:输入特征的个数
hidden_dim:提取的时间点的，论文中的j
kernel_size:卷积核大小，二维，分别为高度和宽度，宽度会决定组合数量
stride:步幅，二维，高度和宽度方向的步幅
'''
class LaxCat(nn.Module):
    def __init__(self, time_num, feature_num, label_num, hidden_dim=32, kernel_size=(3, 3), stride=(1, 1)):
        super(LaxCat, self).__init__()
        le = int((time_num - kernel_size[1]) / stride[1]) + 1
        '''
        二维卷积
        整一个排列索引数组，然后遍历选取
        '''
        feature_index = list(range(feature_num))
        self.combinations = list(itertools.combinations(feature_index, kernel_size[0]))
        '''
        创建组合数个数的卷积层，对每个组合数分别卷积
        '''
        self.Conv2 = nn.ModuleList([
            nn.Conv2d(1, hidden_dim, kernel_size=kernel_size, stride=stride) for _ in range(len(self.combinations))])
        # 变量注意力评分
        self.variable_attn = input_attention_layer(p=len(self.combinations), j=hidden_dim)
        # 时序注意力评分
        self.temporal_attn = temporal_attention_layer(j=hidden_dim, l=le)
        self.fc = nn.Linear(hidden_dim, label_num)

    '''
    前向传播函数
    transpose 将时间和变量的维度交换
    x：样本数*时间点数*特征数
    '''
    def forward(self, x):
        # 增加一个通道数便于卷积
        x = x.unsqueeze(1)
        # 交换特征和时间维度，与论文保持一致
        x = x.transpose(2, 3)
        # x = list(x.split(1, 1))
        conv_outputs = []   # 初始化一个output便于存储
        # 对每个样本组合分别卷积
        for i, indexs in enumerate(self.combinations):
            select_features = x[:, :, indexs, :]
            conv_output = self.Conv2[i](select_features)
            conv_outputs.append(conv_output.unsqueeze(-1))
        # 在最后一个维度拼接输出结果
        # 从样本数*j*1*时间间隔数*组合数
        # 到样本数*组合数*j*时间间隔数
        x = torch.cat(conv_outputs, -1).squeeze(2).permute(0,3,1,2)
        x = self.variable_attn(x)
        x = self.temporal_attn(x)
        return self.fc(x)


def main():
    """
    参数:
    sample_num:样本数
    time_num:每个样本采集的时间点的个数
    feature_num:输入的特征数量
    label_num:标签数
    combo_num:组合数
    :return:各个样本的预测结果
    """
    # 随机生成输入张量
    sample_num = 3
    time_steps = 5
    features = 6
    combo_num = 3
    x_random = torch.rand(sample_num, time_steps, features).cuda()

    laxcat_m = LaxCat(time_num=time_steps, feature_num=features, label_num=4).cuda()
    # 统计模型参数总数量
    total_params = sum(p.numel() for p in laxcat_m .parameters())
    print(f'{total_params:,} total parameters.')
    # 输入张量x传递给laxcat模型
    # x = torch.ones(3, 256, 6).cuda()
    # output = laxcat_m(x)
    # print("0的结果", output)

    # 随机产生的x的结果
    rand_output = laxcat_m(x_random)
    print("随机的结果为", rand_output)


if __name__ == '__main__':
    main()
