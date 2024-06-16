## 检查cuda和cudnn能否正确运行的
# import os
# import torch
#
# # 禁用 cuDNN 加速
# # torch.backends.cudnn.enabled = False
#
# # 检查是否有 CUDA 设备
# if torch.cuda.is_available():
#     device = torch.device('cuda')
#     print("CUDA 加速可用")
# else:
#     device = torch.device('cpu')
#     print("CUDA 加速不可用，使用 CPU")
#
# # 示例张量操作
# x = torch.randn(3, 3).to(device)
# y = torch.randn(3, 3).to(device)
# z = x + y
#
# print(z)
#
# import torch
# import torch.nn as nn
#
# # 检查设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # 创建一个简单的卷积层
# conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1).to(device)
#
# # 创建输入张量
# input_tensor = torch.randn(1, 3, 64, 64).to(device)
#
# # 前向传播
# output = conv(input_tensor)
#
# print("卷积操作成功，cuDNN 正常工作")
#
# ## 检查npy文件
# import os
# import numpy as np
#
#
# path = os.path.join('./dataset', "seizure")
# print(path)
# print(os.path.join(path, 'x_train.npy'))
# # 加载训练和测试数据
# # 对标签数据进行处理
# # 将加载的标签数据转换成了int64，再转化为列表
# x_train = np.load(os.path.join(path, 'x_train.npy'))
# # 查看数据形状和类型
# print("Data shape:", x_train.shape)
# print("Data type:", x_train.dtype)
# print("x_train length", len(x_train))
#
# # 打印部分数据
# print("Data (first 5 elements):", x_train[:5])
#
# # 查看预测结果
#
# y_train = np.load(os.path.join(path, 'y_train.npy'))
# # 查看数据形状和类型
# print("Data shape:", y_train.shape)
# print("Data type:", y_train.dtype)
#
# # 打印部分数据
# print("Data (first 5 elements):", y_train[:5])

#
# # 加载 .npy 文件
# file_path = 'path_to_your_file.npy'  # 将 'path_to_your_file.npy' 替换为你的 .npy 文件路径
# data = np.load(file_path)
#
# # 查看数据形状和类型
# print("Data shape:", data.shape)
# print("Data type:", data.dtype)
#
# # 打印部分数据
# print("Data (first 5 elements):", data[:5])


# import torch

# # 创建一个大小为 (6, 4) 的二维张量
# x = torch.randn(6, 4)
# print(x)
# print(len(x))
# # print(x.unsqueeze(-1))
# # print(x.unsqueeze(-1).shape)
# x_list = x.split(1,-1)
# print(x_list)
# print(x_list[0].shape)
# print(torch.cat(x_list,-1))

# # 在第二维上将张量分割成两份
# x_split = list(torch.split(x, 2, dim=1))
# print(x_split)
# print(len(x_split))
#
# import itertools
#
# # 示例变量列表
# variables = ['a', 'b', 'c', 'd']
#
# # 生成所有可能的组合（选择2个变量的组合）
# combinations = list(itertools.combinations(variables, 2))
# print(combinations)
#
# # 生成所有可能的排列（选择2个变量的排列）
# permutations = list(itertools.permutations(variables, 2))
# print(permutations)

#
# import numpy as np
#
# print(np.random.permutation(9))
