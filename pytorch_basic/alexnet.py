import time
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
from matplotlib import pyplot as plt
"""
实现AlexNet

实现说明:
https://tech.foxrelax.com/ml/modern_cnn/alexnet/
"""


def alexnet():
    """
    实现AlexNet

    >>> x = torch.randn((256, 1, 224, 224))
    >>> net = alexnet()
    >>> assert net(x).shape == (256, 10)

    输入:
    x的形状: (batch_size, 1, 224, 224)

    输出:
    output的形状: (batch_size, 10)
    """

    # Linear Output shape:  torch.Size([1, 4096])
    # ReLU Output shape:  torch.Size([1, 4096])
    # Dropout Output shape:  torch.Size([1, 4096])
    # Linear Output shape:  torch.Size([1, 4096])
    # ReLU Output shape:  torch.Size([1, 4096])
    # Dropout Output shape:  torch.Size([1, 4096])
    # Linear Output shape:  torch.Size([1, 10])
    return nn.Sequential(
        # 这里, 我们使用一个11*11的更大窗口来捕捉对象
        # 同时, 步幅为4, 以减少输出的高度和宽度
        # 另外, 输出通道的数目远大于LeNet
        # output [1, 96, 54, 54]
        nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1),
        # output [1, 96, 54, 54]
        nn.ReLU(),
        # output [1, 96, 26, 26]
        nn.MaxPool2d(kernel_size=3, stride=2),
        # 减小卷积窗口, 使用填充为2来使得输入与输出的高和宽一致, 且增大输出通道数
        # output [1, 256, 26, 26]
        nn.Conv2d(96, 256, kernel_size=5, padding=2),
        # output [1, 256, 26, 26]
        nn.ReLU(),
        # output [1, 256, 12, 12]
        nn.MaxPool2d(kernel_size=3, stride=2),
        # 使用三个连续的卷积层和较小的卷积窗口
        # 除了最后的卷积层, 输出通道的数量进一步增加
        # 在前两个卷积层之后, 汇聚层不用于减少输入的高度和宽度
        # output [1, 384, 12, 12]
        nn.Conv2d(256, 384, kernel_size=3, padding=1),
        # output [1, 384, 12, 12]
        nn.ReLU(),
        # output [1, 384, 12, 12]
        nn.Conv2d(384, 384, kernel_size=3, padding=1),
        # output [1, 384, 12, 12]
        nn.ReLU(),
        # output [1, 256, 12, 12]
        nn.Conv2d(384, 256, kernel_size=3, padding=1),
        # output [1, 256, 12, 12]
        nn.ReLU(),
        # output [1, 256, 5, 5]
        nn.MaxPool2d(kernel_size=3, stride=2),
        # output [1, 6400]
        nn.Flatten(),
        # 这里全连接层的输出数量是LeNet中的好几倍. 使用dropout层来减轻过度拟合
        # output [1, 4096]
        nn.Linear(6400, 4096),
        # output [1, 4096]
        nn.ReLU(),
        # output [1, 4096]
        nn.Dropout(p=0.5),
        # output [1, 4096]
        nn.Linear(4096, 4096),
        # output [1, 4096]
        nn.ReLU(),
        # output [1, 4096]
        nn.Dropout(p=0.5),
        # 最后是输出层. 由于这里使用Fashion-MNIST, 所以用类别数为10, 而非论文中的1000
        # output [1, 10]
        nn.Linear(4096, 10))


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
    train_iter, test_iter = load_data_fashion_mnist(batch_size=256,
                                                    resize=(224, 224))
    net = alexnet()
    kwargs = {
        'num_epochs': 10,
        'loss': nn.CrossEntropyLoss(reduction='mean'),
        'optimizer': torch.optim.SGD(net.parameters(), lr=0.01)
    }
    history = train_gpu(net, train_iter, test_iter, **kwargs)
    plot_history(history)


if __name__ == '__main__':
    run()