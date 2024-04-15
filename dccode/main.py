import argparse
import time
import random
import torch
import torch.nn as nn
from dccode.model.laxcat import LaxCat
from utils import *
from sklearn.metrics import recall_score, f1_score, accuracy_score
# import os
# import numpy as np
# from dccode.model.THAT import HARTrans
# from dccode.model.UniTS import UniTS
# from dccode.model.dual import RFNet
# from dccode.model.mann import MaDNN, MaCNN
# from dccode.model.resnet import ResNet
# from dccode.model.static_UniTS import static_UniTS
# from model import *

# 进行参数配置
def parse_args():
    parser = argparse.ArgumentParser(description='train and test')
    parser.add_argument('--config', default='default', type=str)  # Read UniTS hyperparameters（超参数，但是这个是UniTS模型的）
    # 数据库选择
    parser.add_argument('--dataset', default='opportunity_lc', type=str,
                        choices=['opportunity_lc', 'seizure', 'wifi', 'keti'])
    # 模型选择，之后要删除
    parser.add_argument('--model', default='LaxCat', type=str,
                        choices=['LaxCat', 'UniTS', 'THAT', 'RFNet', 'ResNet', 'MaDNN', 'MaCNN', 'static'])
    # 产生随机数的种子
    parser.add_argument('--seed', default=0, type=int)
    # 日志
    parser.add_argument('--log', default='log', type=str,
                        help="Log directory")

    parser.add_argument('--exp', default='', type=str,
                        choices=['', 'noise', 'missing_data'])
    # 精度，不知道有没有用上
    parser.add_argument('--ratio', default=0.2, type=float)
    # gpu数量
    parser.add_argument('--n_gpu', default=0, type=int)
    # 训练次数
    parser.add_argument('--epochs', default=50, type=int)
    # 学习率
    parser.add_argument('--lr', default=1e-3, type=float)
    # 批量大小
    parser.add_argument('--batch_size', default=64, type=int)
    # 是否存储（不知道干什么）
    parser.add_argument('--save', action='store_true')
    # 仅测试（不知道干什么）
    parser.add_argument('--test_only', action='store_true')
    args = parser.parse_args()
    config = read_config(args.config + '.yaml')
    #如果没有日志的保存地址，则创建日志保存地址
    if not os.path.exists(args.log):
        os.mkdir(args.log)
    args.log_path = os.path.join(args.log, args.dataset)
    #
    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)

    # 指定使用哪一个GPU进行加速，仅当有多核GPU时使用
    torch.cuda.set_device(args.n_gpu)
    # 数据库设置
    # input_size:输入特征的大小
    # input_channel:输入特征的通道数
    # hheads:某个参数，与模型架构相关
    # sensor_axis:传感器轴的数量
    if args.dataset == 'opportunity_lc':
        args.input_size = 256
        args.input_channel = 45
        args.hheads = 9
        args.SENSOR_AXIS = 3
    elif args.dataset == 'seizure':
        args.input_channel = 18
        args.input_size = 256
        args.hheads = 6
        args.SENSOR_AXIS = 1
    elif args.dataset == 'wifi':
        args.input_channel = 180
        args.input_size = 256
        args.batch_size = 16
        args.hheads = 9
        args.SENSOR_AXIS = 3
    elif args.dataset == 'keti':
        args.input_channel = 4
        args.input_size = 256
        args.hheads = 4
        args.SENSOR_AXIS = 1
    # 设置日志存储位置
    args.model_save_path = os.path.join(args.log_path, args.model + '_' + args.config + '.pt')
    return args, config


args, config = parse_args()
# 设置日志记录器的配置，包括设置日志级别、指定输出格式、选择输出位置等
log = set_up_logging(args, config)
args.log = log


# 计算模型性能
def test(model, xtest, ytest):
    # 存储预测标签
    y_pred = []
    # 存储真实标签
    y_true = []
    # 上下文管理器，不会进行自动求导（跟踪张量的计算历史）
    # 节省内存并且加快计算速度
    with torch.no_grad():
        # 模型设置为评估模式
        # 关闭模型中的一些特定于训练的功能
        # 以便进行一致的测试
        model.eval()
        # 循环遍历测试数据，每次处理batch_size个样本
        # 为什么要处理batch_size个样本呢？
        # for i in range(0, len(xtest), args.batch_size):
        for i in range(0, len(xtest), args.batch_size):
            # 如果当前批次大小小于batch_size，则取剩下的样本
            if i + args.batch_size <= len(xtest):
                x = torch.Tensor(xtest[i: i + args.batch_size]).cuda()
                y_true += ytest[i: i + args.batch_size]
            else:
                x = torch.Tensor(xtest[i:]).cuda()
                y_true += ytest[i:]
            # 进行前向传播，得到输出
            out = model(x)
            # 预测结果放入out中
            pred = torch.argmax(out, dim=-1)
            y_pred += pred.cpu().tolist()
    # 显示精确度以及Macro F1分数
    log("Accuracy : " + str(accuracy_score(y_true, y_pred)) +
        "\nMacro F1 : " + str(f1_score(y_true, y_pred, labels=list(range(args.num_labels)), average='macro')))


def main():
    # 打印日志时间
    log("Start time:" + time.asctime(time.localtime(time.time())))
    # 使用read_data函数从args中获得训练样本和测试样本的相关值
    xtrain, ytrain, xtest, ytest = read_data(args, config)
    # 打印使用的模型
    print(args.model)

    model = LaxCat(input_size=args.input_size, input_channel=args.input_channel, num_label=args.num_labels,
                   hidden_dim=64, kernel_size=32, stride=8).cuda()
    # 使用Adam优化器，传入模型参数和学习率
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    #
    total_params = sum(p.numel() for p in model.parameters())
    log('Total parameters: ' + str(total_params))

    # 如果配置的是测试环境，只对模型进行测试，不进行训练
    if args.test_only:
        if os.path.exists(args.model_save_path):
            model.load_state_dict(torch.load(args.model_save_path))
            test(model, xtest, ytest)
        else:
            log("Model state dict not found!")
        return

    # 训练数据的洗牌
    random.seed(args.seed)
    random.shuffle(xtrain)
    random.seed(args.seed)
    random.shuffle(ytrain)

    # 定义损失函数为交叉熵损失
    loss_func = nn.CrossEntropyLoss()
    try:
        # 开始迭代运行模型
        for ep in range(1, 1 + args.epochs):
            # 开始训练
            model.train()
            # 定义迭代损失
            epoch_loss = 0

            log("Training epoch : " + str(ep))
            # for i in range(0, len(xtrain), args.batch_size):
            #     # 将输入特征和标签转化为PyTorch张量，并且转移到GPU上
            #     if i + args.batch_size <= len(xtrain):
            #         x = torch.from_numpy(np.array(xtrain[i: i + args.batch_size])).cuda()
            #         x = torch.tensor(x, dtype=torch.float32).cuda()
            #         # x = torch.Tensor(xtrain[i: i + args.batch_size]).cuda()
            #         y = torch.LongTensor(ytrain[i: i + args.batch_size]).cuda()
            #     else:
            #         x = torch.Tensor(xtrain[i:]).cuda()
            #         y = torch.LongTensor(ytrain[i:]).cuda()
            #     # 得到输出和loss
            #     out = model(x)
            #     loss = loss_func(out, y)
            #     epoch_loss += loss.cpu().item()
            #     # 执行反向传播，和参数更新
            #     optimizer.zero_grad()
            #     loss.backward()
            #     optimizer.step()
            for i in range(0, len(xtrain), args.batch_size):
                # 将输入特征和标签转化为PyTorch张量，并且转移到GPU上
                if i + args.batch_size <= len(xtrain):
                    x = torch.tensor(torch.from_numpy(np.array(xtrain[i: i + args.batch_size])), dtype=torch.float32).cuda()
                    # x = torch.Tensor(xtrain[i: i + args.batch_size]).cuda()
                    y = torch.LongTensor(ytrain[i: i + args.batch_size]).cuda()
                else:
                    x = torch.Tensor(xtrain[i:]).cuda()
                    y = torch.LongTensor(ytrain[i:]).cuda()
                # 得到输出和loss
                out = model(x)
                loss = loss_func(out, y)
                epoch_loss += loss.cpu().item()
                # 执行反向传播，和参数更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # 打印每个epoch的训练损失
            log("Training loss : " + str(epoch_loss / (i / args.batch_size + 1)))
            # 在测试集上对模型进行测试，并且打印结果
            test(model, xtest, ytest)
            log("----------------------------")
    # 如果收到键盘中断信号，则停止
    except KeyboardInterrupt:
        print('Exiting from training early')
        test(model, xtest, ytest)
    # 如果设置了保存路径则存储模型状态字典到指定路径
    if args.save:
        torch.save(model.state_dict(), args.model_save_path)


if __name__ == '__main__':
    main()