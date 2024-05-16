import torch
import matplotlib.pyplot as plt

# # 创建二维数据
# data = torch.rand(10, 10)
# print(data)
# # 绘制色块图
# # cmap 是 对印的颜色映射
# # interpolation指定了插值方法:最邻近插值
# plt.imshow(data, cmap='viridis', interpolation='nearest')
# plt.colorbar()  # 添加颜色条
# plt.show()


# 点乘测试
x = torch.tensor([[1,2],[3,4]])
y = torch.tensor([[1],[2]])
print(torch.mul(x,y))