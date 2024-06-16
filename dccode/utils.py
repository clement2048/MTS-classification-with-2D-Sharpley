import os
import yaml
import numpy as np
import torch
# 工具包


# 读取数据集
def read_data(args, config):
    # 构建数据集路径
    path = os.path.join('./dataset', args.dataset)
    print(path)
    print(os.path.join(path, 'x_train.npy'))

    # 加载训练和测试数据
    x_train = torch.tensor(np.load(os.path.join(path, 'x_train.npy')), dtype=torch.float32)
    y_train = torch.tensor(np.load(os.path.join(path, 'y_train.npy')), dtype=torch.long)
    x_test = torch.tensor(np.load(os.path.join(path, 'x_test.npy')), dtype=torch.float32)
    y_test = torch.tensor(np.load(os.path.join(path, 'y_test.npy')), dtype=torch.long)

    np.random.seed(args.seed)

    # 鲁棒性测试（噪声测试）
    if args.exp == 'noise':
        for i in range(len(x_train)):
            for j in range(x_train.shape[2]):
                noise = np.random.normal(1, 1, size=x_train[i][:, j].shape)
                x_train[i][:, j] = x_train[i][:, j] + noise * args.ratio * np.mean(np.absolute(x_train[i][:, j]))
        for i in range(len(x_test)):
            for j in range(x_test.shape[2]):
                noise = np.random.normal(1, 1, size=x_test[i][:, j].shape)
                x_test[i][:, j] = x_test[i][:, j] + noise * args.ratio * np.mean(np.absolute(x_test[i][:, j]))

    # 鲁棒性测试（缺失值测试）
    elif args.exp == 'missing_data':
        for i in range(len(x_train)):
            for j in range(x_train.shape[2]):
                mask = np.random.random(x_train[i][:, j].shape) >= args.ratio
                x_train[i][:, j] = x_train[i][:, j] * mask
        for i in range(len(x_test)):
            for j in range(x_test.shape[2]):
                mask = np.random.random(x_test[i][:, j].shape) >= args.ratio
                x_test[i][:, j] = x_test[i][:, j] * mask

    # 统计标签类别数量
    args.num_labels = len(set(y_train.tolist()))
    summary = [0 for i in range(args.num_labels)]
    for i in y_train:
        summary[i] += 1

    # 打印标签数量和数据集大小信息
    print("Label num cnt:", summary)
    print("Training size:", len(y_train))
    print("Testing size:", len(y_test))

    return x_train, y_train, x_test, y_test

    # x_train = np.load(os.path.join(path, 'x_train.npy'))
    # y_train = np.load(os.path.join(path, 'y_train.npy')).astype('int64').tolist()
    # x_test = np.load(os.path.join(path, 'x_test.npy'))
    # y_test = np.load(os.path.join(path, 'y_test.npy')).astype('int64').tolist()


# 定义AttrDict类
# 将字典的键值对 转化为 类的属性，使得可以通过属性访问字典中的值
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


# 读取配置
def read_config(path):
    return AttrDict(yaml.load(open(path, 'r'), Loader=yaml.FullLoader))


def logging(file):
    def write_log(s):
        print(s)
        with open(file, 'a') as f:
            f.write(s + '\n')

    return write_log


def set_up_logging(args, config):
    log = logging(os.path.join(args.log_path, args.model + '.txt'))
    for k, v in config.items():
        log("%s:\t%s\n" % (str(k), str(v)))
    return log
