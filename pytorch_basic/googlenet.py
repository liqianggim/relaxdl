import time
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
from matplotlib import pyplot as plt
import torch.nn.functional as F
"""
实现GoogLeNet

实现说明:
https://tech.foxrelax.com/ml/modern_cnn/googlenet/
"""

class Inception(nn.Module):
    # `c1`--`c4` 是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super().__init__()
        # 线路1, 单1 x 1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2, 1 x 1卷积层后接3 x 3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3, 1 x 1卷积层后接5 x 5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4, 3 x 3最大汇聚层后接1 x 1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, X):
        p1 = F.relu(self.p1_1(X))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(X))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(X))))
        p4 = F.relu(self.p4_2(self.p4_1(X)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)


def googlenet():
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                       nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
                       nn.Conv2d(64, 192, kernel_size=3, padding=1),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                       Inception(256, 128, (128, 192), (32, 96), 64),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                       Inception(512, 160, (112, 224), (24, 64), 64),
                       Inception(512, 128, (128, 256), (24, 64), 64),
                       Inception(512, 112, (144, 288), (32, 64), 64),
                       Inception(528, 256, (160, 320), (32, 128), 128),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                       Inception(832, 384, (192, 384), (48, 128), 128),
                       nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
    return nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))


def load_data_fashion_mnist(batch_size, resize=None, root='../data'):
    """
    下载Fashion-MNIST数据集, 然后将其加载到内存中

    1. 60000张训练图像和对应Label
    2. 10000张测试图像和对应Label
    3. 10个类别
    4. 每张图像28x28x1的分辨率

    >>> train_iter, test_iter = load_data_fashion_mnist(batch_size=256)
    >>> for x, y in train_iter:
    >>>     assert x.shape == (256, 1, 28, 28)
    >>>     assert y.shape == (256, )
    >>>     break
    """
    # 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式
    # 并除以255使得所有像素的数值均在0到1之间
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root,
                                                    train=True,
                                                    transform=trans,
                                                    download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root=root,
                                                   train=False,
                                                   transform=trans,
                                                   download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True),
            data.DataLoader(mnist_test, batch_size, shuffle=False))


def accuracy(y_hat, y):
    """
    计算预测正确的数量

    参数:
    y_hat的形状: (batch_size, num_classes)
    y的形状: (batch_size, )
    """
    _, predicted = torch.max(y_hat, 1)
    cmp = predicted.type(y.dtype) == y
    return cmp.type(y.dtype).sum()


def train_gpu(net,
              train_iter,
              test_iter,
              num_epochs=10,
              loss=None,
              optimizer=None,
              device=None,
              verbose=False):
    """
    用GPU训练模型
    """
    if device is None:
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    if loss is None:
        loss = nn.CrossEntropyLoss(reduction='mean')
    if optimizer is None:
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    times = []
    history = [[], [], []]  # 记录: 训练集损失, 训练集准确率, 测试集准确率, 方便后续绘图
    num_batches = len(train_iter)
    for epoch in range(num_epochs):
        # 训练
        metric_train = [0.0] * 3  # 统计: 训练集损失之和, 训练集准确数量之和, 训练集样本数量之和
        net.train()
        for i, (X, y) in enumerate(train_iter):
            t_start = time.time()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric_train[0] += float(l * X.shape[0])
                metric_train[1] += float(accuracy(y_hat, y))
                metric_train[2] += float(X.shape[0])
            times.append(time.time() - t_start)
            train_loss = metric_train[0] / metric_train[2]
            train_acc = metric_train[1] / metric_train[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                if verbose:
                    print(
                        f'epoch {epoch}, step {i+1}, train loss {train_loss:.3f}, '
                        f'train acc {train_acc:.3f}')
                history[0].append((epoch + (i + 1) / num_batches, train_loss))
                history[1].append((epoch + (i + 1) / num_batches, train_acc))

        # 评估
        metric_test = [0.0] * 2  # 测试准确数量之和, 测试样本数量之和
        net.eval()
        with torch.no_grad():
            for X, y in test_iter:
                X, y = X.to(device), y.to(device)
                metric_test[0] += float(accuracy(net(X), y))
                metric_test[1] += float(X.shape[0])
            test_acc = metric_test[0] / metric_test[1]
            history[2].append((epoch + 1, test_acc))
            print(f'epoch {epoch}, step {i+1}, train loss {train_loss:.3f}, '
                  f'train acc {train_acc:.3f}, test acc {test_acc:.3f}')

    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric_train[2] * num_epochs / sum(times):.1f} '
          f'examples/sec on {str(device)}')
    return history


def plot_history(history, figsize=(6, 4)):
    plt.figure(figsize=figsize)
    # 训练集损失, 训练集准确率, 测试集准确率
    num_epochs = len(history[2])
    plt.plot(*zip(*history[0]), '-', label='train loss')
    plt.plot(*zip(*history[1]), 'm--', label='train acc')
    plt.plot(*zip(*history[2]), 'g-.', label='test acc')
    plt.xlabel('epoch')
    # 从epoch=1开始显示, 0-1这个范围的数据丢弃不展示,
    # 因为只有训练完成1个epochs之后, 才会有第一条test acc记录
    plt.xlim((1, num_epochs))
    plt.grid()
    plt.legend()
    plt.show()


def run():
    train_iter, test_iter = load_data_fashion_mnist(batch_size=128,
                                                    resize=(96, 96))
    net = googlenet()
    kwargs = {
        'num_epochs': 10,
        'loss': nn.CrossEntropyLoss(reduction='mean'),
        'optimizer': torch.optim.SGD(net.parameters(), lr=0.1)
    }
    history = train_gpu(net, train_iter, test_iter, **kwargs)
    plot_history(history)


if __name__ == '__main__':
    run()