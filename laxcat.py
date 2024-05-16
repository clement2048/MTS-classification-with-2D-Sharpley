import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# 输入注意力层
# 继承与nn.Module
class input_attention_layer(nn.Module):
    # p:输入数据的通道数，
    # j:隐藏单元的维度
    def __init__(self, p, j):
        super(input_attention_layer, self).__init__()
        self.weight1 = nn.Parameter(torch.ones(1, p), requires_grad=True)
        self.bias1 = nn.Parameter(torch.zeros(1, j), requires_grad=True)
        self.weight2 = nn.Parameter(torch.ones(j, p), requires_grad=True)
        self.bias2 = nn.Parameter(torch.zeros(1, p), requires_grad=True)

    # 返回两个参数
    # 加权后的变量和注意力矩阵
    def forward(self, x):
        # x: B * l * p * j
        v0 = x.size(0)
        v1 = x.size(1)
        v3 = x.size(3)
        a = torch.zeros(v0, v1, v3).cuda()
        # print("input attention x shape is ", x.shape)
        # print(x[0, 0, :, :].shape)
        # 用A保存注意力系数
        A = torch.zeros(v0, v1, x.size(2)).cuda()
        for i in range(v0):
            h = torch.zeros(v1, v3)
            for j in range(v1):
                # F 是 torch.nn.functional 模块的别名
                # x: B * l * p * j
                xtmp = x[i, j, :, :]
                tmp = F.sigmoid(torch.matmul(self.weight1, xtmp) + self.bias1)
                tmp = F.sigmoid(torch.matmul(tmp, self.weight2) + self.bias2)
                # tmp: 1 * p, p个变量的分数
                attn = F.softmax(tmp, -1)
                A[i][j] = attn[0]
                h[j] = torch.matmul(attn, x[i, j, :, :])
            a[i] = h
        return a, A
        # B * l * j


# 时序注意力层
class temporal_attention_layer(nn.Module):
    def __init__(self, j, l):
        super(temporal_attention_layer, self).__init__()
        # 设置参数
        # 这些参数的维度都是论文上的
        self.weight1 = nn.Parameter(torch.ones(1, l), requires_grad=True)
        self.bias1 = nn.Parameter(torch.zeros(1, j), requires_grad=True)
        self.weight2 = nn.Parameter(torch.ones(j, l), requires_grad=True)
        self.bias2 = nn.Parameter(torch.zeros(1, l), requires_grad=True)

    # 前向传播
    def forward(self, x):
        # x: 样本数 * l * j
        # 这里使用ReLU作为非线性激活函数
        v0 = x.size(0)
        h = torch.zeros(v0, x.size(2)).cuda()
        b = torch.zeros(v0, x.size(1)).cuda()
        for i in range(v0):
            tmp = F.sigmoid(torch.matmul(self.weight1, x[i, :, :]) + self.bias1)
            tmp = F.sigmoid(torch.matmul(tmp, self.weight2) + self.bias2)
            # attn: 1 * l
            attn = F.softmax(tmp/math.sqrt(tmp.size(1)), -1)
            b[i] = attn[0]
            tempa = torch.matmul(attn, x[i, :, :])
            h[i] = tempa[0]
        return h, b


# 对于时序处理问题，输入通道数和输出通道数怎么确定
# 分类层
class LaxCat(nn.Module):
    def __init__(self, tp_num, var_num, num_label, J=2, L=4, stride=3):
        super(LaxCat, self).__init__()
        # tp_num(time_point_num),时间点个数，即采样个数
        # var_num:变量个数,采样的维度
        # stride : 步长
        # 通过公式会得到提取特征后的样本个数
        l = int((tp_num - L) / stride) + 1
        # 卷积特征提取层
        self.Conv1 = nn.ModuleList([
            nn.Conv1d(1, J, kernel_size=L, stride=stride) for _ in range(var_num)])
        # 变量注意力评分
        self.variable_attn = input_attention_layer(p=var_num, j=J)
        # 时序注意力评分
        self.temporal_attn = temporal_attention_layer(j=J, l=l)
        # 参数分别为输入参数数量和输出参数数量
        self.fc = nn.Linear(J, num_label)

    # 前向传播函数
    def forward(self, x):
        # 对输入张量进行维度转换
        # 将第一维和第二维进行转置，常用于处理时序数据
        # 将时间步和通道维度转换
        # 从数据个数*时间点数*变量个数
        # 到数据个数*变量个数*时间点数，
        # 便于用1*L提取变量
        x = x.transpose(1, 2)
        x = list(x.split(1, 1))
        # 对每个变量
        for i in range(len(x)):
            x[i] = self.Conv1[i](x[i]).unsqueeze(-1)
        # cat将经过卷积的子张量沿着最后一维拼接
        # 通过permute重新排列维度顺序
        # 以便后续注意力机制或全连接层处理
        x = (torch.cat(x, -1))
        # x:样本数 * J * l * P
        # permute可以转化维度
        x = x.permute(0, 2, 3, 1)
        # x:样本数 * L * P * J
        v0 = x.size(0)
        attn = torch.zeros(v0, x.size(1), x.size(2)).cuda()
        # x = F.relu(self.Conv1(x)).reshape(B, C, -1, x.size(0))
        outv, A = self.variable_attn(x)
        outt, b = self.temporal_attn(outv)
        # 计算joint_attn
        for i in range(v0):
            tattn = b[i, :]
            tattn = torch.transpose(torch.unsqueeze(tattn, 0),0,1)
            tmp = torch.mul(tattn, A[i, :, :])
            attn[i] = tmp
        return self.fc(outt), attn


# 测试代码
# 样本个数
# 时间节点
# 变量数
def main():
    # 创建模型并且转移到GPU上
    stft_m = LaxCat(tp_num=16, var_num=2, num_label=6).cuda()
    # 统计模型参数总数量
    y = torch.rand(3).cuda()
    y = torch.tensor(y * 6, dtype=int)
    total_params = sum(p.numel() for p in stft_m.parameters())
    print(f'{total_params:,} total parameters.')
    # 输入张量x传递给模型‘stft_m’,并且前向传播，将输出结果
    x = torch.rand(3, 16, 2).cuda()
    # 张量x传递给模型进行前向传播，打印输出
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(stft_m.parameters(), lr=0.05)
    # 定义迭代损失
    for i in range(5):
        epoch_loss = 0
        output, out_attn = stft_m(x)
        print("output is", output)
        print("out_attn is", out_attn)
        loss = loss_fn(output, y)
        epoch_loss += loss.cpu().item()
        print("loss is", epoch_loss)
        print("output attn is:",out_attn)
        # 查看准确率
        # for j in range()
        # 执行反向传播，和参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 打印每个epoch的训练损失


if __name__ == '__main__':
    main()
