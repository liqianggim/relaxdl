import os
import hashlib
import requests
from torch import nn
import torch
from torch import Tensor
"""
实现MobileNet V2

实现说明:
https://tech.foxrelax.com/ml/lightweight/mobilenet_v2/
"""


def _make_divisible(ch: float, divisor: int = 8, min_ch: int = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


def load_weight(cache_dir: str = '../data') -> str:
    """
    加载预训练权重(class=1000的ImageNet数据集上训练的)
    """
    sha1_hash = '9d6df55a618d1707f020679b8cd68c91d4dec003'
    url = 'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/mobilenet_v2-b0353104.pth'
    fname = os.path.join(cache_dir, url.split('/ml/')[-1])
    fdir = os.path.dirname(fname)
    os.makedirs(fdir, exist_ok=True)
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    print(f'download {url} -> {fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    print(f'download {fname} success!')
    # e.g. ../data/mobilenet_v2-b0353104.pth
    return fname


class ConvBNReLU(nn.Sequential):
    """
    1. 当groups=1的时候是普通卷积
    2. 当groups=in_channel=out_channel时, 是Depthwise Separable卷积(DW)
    """
    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1) -> None:
        """
        参数:
        in_channel: 输入通道数
        out_channel: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长
        groups: 1或者in_channel
        """
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel,
                      out_channel,
                      kernel_size,
                      stride,
                      padding,
                      groups=groups,
                      bias=False), nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True))


class InvertedResidual(nn.Module):
    """
    实现Inverted Residual Block

    注意: 
    当stride=1并且in_channel == out_channel会有残差连接, 否则没有残差连接
    因为stride不等于1等于做了降维操作, 没法直接做add; 输入通道和输出通道不一致, 
    也没法直接做add
    """
    def __init__(self, in_channel: int, out_channel: int, stride: int,
                 expand_ratio: float) -> None:
        """
        参数:
        in_channel: 输入通道数
        out_channel: 输出通道数
        stride: 步长
        expand_ratio: 隐藏层的扩展因子
        """
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio
        # 判断是否有残差连接
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv
            layers.append(ConvBNReLU(in_channel, hidden_channel,
                                     kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel,
                       hidden_channel,
                       stride=stride,
                       groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            # 注意: 这里是没有ReLU6激活函数的
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes: int = 1000,
                 alpha: float = 1.0,
                 round_nearest: int = 8) -> None:
        """
        参数:
        num_classes: 分类数量
        alpha: 模型缩放因子
               当alpha>1时, 相当于扩大模型的规模
               alpha<1时, 相当于缩小模型的规模
        round_nearest: 默认为8
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        # 根据MobileNet V2论文中的设置配置倒残差块
        inverted_residual_setting = [
            # t, c, n, s
            # t: 膨胀因子, 也就是每个倒残差块用1x1卷积升维之后的通道数
            # c: 输出通道数
            # n: 重复几次
            # s: 第一个倒残差快的stirde
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        # conv1 layer
        features.append(ConvBNReLU(3, input_channel, stride=2))
        # 构建: Inverted Residual Blockes
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                # stride只有在重复第一次的时候为s, 其它时候都为1,
                # 因为我们的降维操作只做一次
                stride = s if i == 0 else 1
                features.append(
                    block(input_channel,
                          output_channel,
                          stride,
                          expand_ratio=t))
                input_channel = output_channel
        features.append(ConvBNReLU(input_channel, last_channel, 1))
        self.features = nn.Sequential(*features)

        # 构建分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Dropout(0.2),
                                        nn.Linear(last_channel, num_classes))

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def mobilenet_v2(pretrained: bool = True,
                 num_classes: int = 5,
                 alpha: float = 1.0,
                 round_nearest: float = 8) -> MobileNetV2:
    net = MobileNetV2(num_classes, alpha, round_nearest)
    if pretrained:
        model_weight_path = load_weight(cache_dir='../data')
        pre_weights = torch.load(model_weight_path, map_location='cpu')
        # 当新构建的网络(net)的分类器的数量和预训练权重分类器的数量不一致时, 删除分类器这一层的权重
        pre_dict = {
            k: v
            for k, v in pre_weights.items()
            if net.state_dict()[k].numel() == v.numel()  # 只保留权重数量一致的层
        }
        # 加载权重
        net.load_state_dict(pre_dict, strict=False)
    return net