from typing import List, Optional
import os
import hashlib
import requests
import torch.nn as nn
import torch
from torch import Tensor


def load_weight(cache_dir: str = '../data') -> str:
    """
    加载预训练权重(class=1000的ImageNet数据集上训练的)
    """
    sha1_hash = '6ba9789036078cf8bace8dd75a770f46789c350c'
    url = 'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/resnet50-0676ba61.pth'
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
    # e.g. ../data/resnet50-0676ba61.pth
    return fname


class Bottleneck(nn.Module):
    """
    1. 输入的通道数是: in_channel
    2. 最终输出的通道数是: out_channel*expansion
    3. 如果stride=1, 则最终输出的h/w不变;如果stride=2, 则最终输出的h/w减半
    4. 在下面两种情况下需要downsample(x)来生成identity
       <1> 如果stride != 1, 最终输出的h/w会减半
       <2> 如果out_channel*expansion != in_channel, 最终输出的通道数会改变
    
    核心思想:
    分支1: x -> downsample(可选) -> identity
    分支2: x -> 1x1(降维 squeeze channels) -> 3x3(维度不变, stride=1|2) -> 1x1(升维 unsqueeze channels) -> out
    out = out+identity
    """
    expansion = 4

    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None) -> None:
        super(Bottleneck, self).__init__()
        """
        参数:
        in_channel: 输入的通道数
        out_channel: 中间3x3卷积的输入和输出通道数
        stride: 中间3x3卷积的stride
                如果stride=1, 则最终输出的h/w不变;
                如果stride=2, 则最终输出的h/w减半
        downsample: x -> downsample(x) -> identity
                a. 如果stride != 1, 最终输出的h/w会减半
                b. 如果out_channel*expansion != in_channel, 最终输出的通道数会改变
                在这两种情况下, 需要downsample(x)来生成identity
        """
        # 1x1
        self.conv1 = nn.Conv2d(in_channels=in_channel,
                               out_channels=out_channel,
                               kernel_size=1,
                               stride=1,
                               bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        # 3x3
        self.conv2 = nn.Conv2d(
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=stride,  # 是否改变h/w
            bias=False,
            padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # 1x1
        self.conv3 = nn.Conv2d(in_channels=out_channel,
                               out_channels=out_channel * self.expansion,
                               kernel_size=1,
                               stride=1,
                               bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,
                 block: Bottleneck,
                 blocks_num: List[int],
                 num_classes=1000,
                 include_top=True) -> None:
        """
        参数:
        block: Bottleneck
        blocks_num: layers的配置, 每一层的block数量
        num_classes: 分类数量
        include_top: 是否返回最终的全连接层
        """
        # 我们假设输入的shape为: [3, 224, 224]
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        # 7x7
        # [3, 224, 224] -> [64, 112, 112]
        self.conv1 = nn.Conv2d(3,
                               self.in_channel,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        # 3x3
        # [64, 112, 112] -> [64, 56, 56]
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # [64, 56, 56] -> [256, 56, 56]
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        # [256, 56, 56] -> [512, 28, 28]
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        # [512, 28, 28] -> [1024, 14, 14]
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        # [1024, 14, 14] -> [2048, 7, 7]
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            # [2048, 7, 7] -> [2048, 1, 1]
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            # [2048, ] -> [1000, ]
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')

    def _make_layer(self,
                    block: nn.Module,
                    channel: int,
                    block_num: int,
                    stride: int = 1) -> nn.Module:
        """
        构建一层, 会有`block_num`个block
        1. 会根据前一层自动计算输入通道
        2. 最终输出的通道数是: channel*expansion
        3. 如果stride=1, 则最终输出的h/w不变;如果stride=2, 则最终输出的h/w减半

        参数:
        block: Bottleneck
        channel: block的参数-输入的通道数
        block_num: 构建block的数量
        stride: block的参数-stride
                如果stride=1, 则block最终输出的h/w不变
                如果stride=2, 则block最终输出的h/w减半
        """
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel,
                          channel * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        # 只有第1个block会使用stride参数来改变h/w
        layers = []
        layers.append(
            block(self.in_channel,
                  channel,
                  downsample=downsample,
                  stride=stride))
        self.in_channel = channel * block.expansion

        # 其它所有block的stride=1
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        >>> net = resnet_50(include_top=False)
        >>> x = torch.randn((12, 3, 224, 224))
        >>> assert net(x).shape == (12, 2048, 7, 7)

        返回:
        1. 如果需要include_top=True, 返回: [batch_size, 2048, h, w]
           下采样率为32, 也就是如果原始输入尺寸为: [batch_size, 3, 224, 224]
           最终输出尺寸为: [batch_size, 2048, 224/32, 224/32] = [batch_size, 2048, 7, 7]
        2. 如果需要include_top=False, 返回: [batch_size, num_classes]
        """
        # 我们假设输入的shape为: [batch_size, 3, 224, 224], num_classes=1000
        # [batch_size, 3, 224, 224] -> [batch_size, 64, 112, 112]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # [batch_size, 64, 112, 112] -> [batch_size, 64, 56, 56]
        x = self.maxpool(x)

        # [batch_size, 64, 56, 56] -> [batch_size, 256, 56, 56]
        x = self.layer1(x)
        # [batch_size, 256, 56, 56] -> [batch_size, 512, 28, 28]
        x = self.layer2(x)
        # [batch_size, 512, 28, 28] -> [batch_size, 1024, 14, 14]
        x = self.layer3(x)
        # [batch_size, 1024, 14, 14] -> [batch_size, 2048, 7, 7]
        x = self.layer4(x)

        if self.include_top:
            # [batch_size, 2048, 7, 7] -> [batch_size, 2048, 1, 1]
            x = self.avgpool(x)
            # [batch_size, 2048, 1, 1] -> [batch_size, 2048]
            x = torch.flatten(x, 1)
            # [batch_size, 2048, 1, 1] -> [batch_size, 1000]
            x = self.fc(x)

        return x


def resnet_50(pretrained: bool = True,
              num_classes: int = 1000,
              include_top: bool = True) -> ResNet:
    net = ResNet(Bottleneck, [3, 4, 6, 3],
                 num_classes=num_classes,
                 include_top=include_top)
    if pretrained:
        model_weight_path = load_weight(cache_dir='../data')
        pre_weights = torch.load(model_weight_path, map_location='cpu')
        # 当新构建的网络(net)的分类器的数量和预训练权重分类器的数量不一致时, 删除分类器这一层的权重
        if include_top:
            pre_dict = {
                k: v
                for k, v in pre_weights.items()
                if net.state_dict()[k].numel() == v.numel()  # 只保留权重数量一致的层
            }
        else:
            pre_dict = pre_weights
        # 加载权重
        net.load_state_dict(pre_dict, strict=False)
    return net
