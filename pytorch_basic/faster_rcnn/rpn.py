from tkinter import Image
from typing import List, Optional, Dict, Tuple
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from transform import ImageList
from box import box_iou, clip_boxes_to_image, remove_small_boxes, batched_nms
from det_utils import BoxCoder, Matcher, BalancedPositiveNegativeSampler, smooth_l1_loss


class AnchorsGenerator(nn.Module):
    __annotations__ = {
        "cell_anchors": Optional[List[Tensor]],
        "_cache": Dict[str, List[Tensor]]
    }
    """
    将`一个批量的图片imageList`以及`其backbone输出的feature_maps`作为输入, 输出
    `每张图片对应的anchors(是由feature map上的anchors映射过去的)`

    1. 该模块可以针对多层features map来生成anchors
    2. 不同的feature map可以有不同的sizes & aspect_ratios配置
       <1> sizes[i]和aspect_ratios[i]表示第i层的feature map对应的anchors配置
       <2> 每一层feature map的每个像素会生成的anchors数量是: len(sizes[i]) * len(aspect_ratios[i])

    下面的例子中我们可以看到每个图片会输出19890个anchors, 19890=34x39x15, 
    其中(34, 39)是feature map的h/w, 15是feature map的每个像素生成的anchors的数量
    >>> backbone = create_backbone(num_classes=21)
    >>> anchor_generator = AnchorsGenerator()
    >>> batch_size = 8
    >>> train_iter, _ = load_pascal_voc(batch_size)
    >>> for images, targets in train_iter:
    >>>     images = list(image.to(device) for image in images)
    >>>     targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    >>>     images, targets = transform(images, targets)
    >>>     break
    >>> features = backbone(images.tensors)
    >>> print(features.shape)
        torch.Size([8, 1280, 34, 39])
    
    >>> # 若backbone只生成一层特征层图, 将feature放入有序字典中, 并编号为'0'
    >>> # 若backbone生成多层特征图, 传入的就是一个有序字典
    >>> # 我们当前的backbone生成的只有一层特征图
    >>> if isinstance(features, torch.Tensor):
    >>>     features = OrderedDict([('0', features)])
    >>> anchors = anchor_generator(images, list(features.values()))
    >>> for anchor_per_image in anchors:
    >>>     print(anchor_per_image.shape)
        torch.Size([19890, 4])
        torch.Size([19890, 4])
        torch.Size([19890, 4])
        torch.Size([19890, 4])
        torch.Size([19890, 4])
        torch.Size([19890, 4])
        torch.Size([19890, 4])
        torch.Size([19890, 4])
    """
    def __init__(
        self,
        sizes: Tuple[Tuple[int]] = ((32, 64, 128, 256, 512), ),
        aspect_ratios: Tuple[Tuple[float]] = ((0.5, 1.0, 2.0), )
    ) -> None:
        """
        有几层feature map, sizes/aspect_ratios参数就有几个Tuple, 默认参数是针对
        MobileNetV2设置的, 因为其作为backbone只输出了一层feature map, 所以参数只有一个Tuple

        aspect_ratios表示高宽比, 如果: aspect_ratio=2, size=512, 则返回的anchor:
        高度为:sqrt{2} * 512
        宽度为:(1/sqrt{2}) * 512

        参数:
        sizes: Tuple[Tuple[int]], 尺寸. 有几层feature map, 就有几个Tuple
        aspect_ratios: Tuple[Tuple[float]], 高宽比. 有几层feature map, 就有几个Tuple
        """
        super(AnchorsGenerator, self).__init__()
        assert len(sizes) == len(aspect_ratios)

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        # 保存了每一层feature map对应的anchors模板(也就是feature map的每个像素如何生成anchors):
        # cell_anchors的形状: List[Tensor]
        # <1> Tensor的形状: (num_anchors, 4), 不同feature map的num_anchors是不同的
        # <2> num_anchors = len(sizes)*len(aspect_ratios)
        #     比如某一层feature map的: sizes=(32, 64, 128, 256, 512),
        #     aspect_ratios=(0.5, 1.0, 2.0), 则num_anchors=15
        self.cell_anchors = None
        self._cache = {}

    def generate_anchors(
        self,
        scales: Tuple[int],
        aspect_ratios: Tuple[float],
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device('cpu')
    ) -> Tensor:
        """
        针对一层feature map, 生成其对应的anchors模板(也就是feature map的每个像素如何生成anchors)
        每个anchor中心点的坐标为(0,0), 返回的是其左上角的坐标和右下角的坐标

        >>> gen = AnchorsGenerator()
        >>> scales = (32, 64, 128, 256, 512)
        >>> aspect_ratios = (0.5, 1.0, 2.0)
        >>> anchors = gen.generate_anchors(scales, aspect_ratios)
        >>> assert anchors.shape == (5 * 3, 4)
        >>> print(anchors)
        tensor([[ -23.,  -11.,   23.,   11.],
                [ -45.,  -23.,   45.,   23.],
                [ -91.,  -45.,   91.,   45.],
                [-181.,  -91.,  181.,   91.],
                [-362., -181.,  362.,  181.],
                [ -16.,  -16.,   16.,   16.],
                [ -32.,  -32.,   32.,   32.],
                [ -64.,  -64.,   64.,   64.],
                [-128., -128.,  128.,  128.],
                [-256., -256.,  256.,  256.],
                [ -11.,  -23.,   11.,   23.],
                [ -23.,  -45.,   23.,   45.],
                [ -45.,  -91.,   45.,   91.],
                [ -91., -181.,   91.,  181.],
                [-181., -362.,  181.,  362.]])

        参数: 
        scales: 某一层feature map的scales, e.g. (32, 64, 128, 256, 512)
        aspect_ratios: 某一层feature map的aspect_ratios, e.g. (0.5, 1.0, 2.0)
        dtype: 返回Tensor的类型
        device: 返回Tensor的设备

        返回:
        anchors的形状: (num_anchors, 4)
         <1> num_anchors = len(scales)*len(aspect_ratios)
        """
        # scales的形状: (num_scales, )
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        # aspect_ratios的形状: (num_aspect_ratios, )
        aspect_ratios = torch.as_tensor(aspect_ratios,
                                        dtype=dtype,
                                        device=device)
        # h_ratios的形状: (num_aspect_ratios, )
        h_ratios = torch.sqrt(aspect_ratios)
        # w_ratios的形状: (num_aspect_ratios, )
        w_ratios = 1.0 / h_ratios

        # [r1, r2, r3] * [s1, s2]
        # ws的形状: (num_aspect_ratios, num_scales) -> (num_aspect_ratios*num_scales, )
        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        # hs的形状: (num_aspect_ratios, num_scales) -> (num_aspect_ratios*num_scales, )
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        # 每个anchor中心点的坐标为(0,0), 返回的是其左上角的坐标和右下角的坐标
        # 生成的anchors模板都是以(0, 0)为中心的, 形状是: (num_anchors, 4)
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2

        return base_anchors.round()  # round四舍五入

    def set_cell_anchors(self, dtype: torch.dtype,
                         device: torch.device) -> None:
        """
        计算每一层feature map的anchors模板(也就是feature map的每个像素如何生成anchors)
        """
        if self.cell_anchors is not None:
            cell_anchors = self.cell_anchors
            assert cell_anchors is not None
            if cell_anchors[0].device == device:
                return

        # 遍历每一层feature map的配置:
        cell_anchors = [
            # 针对一层feature map, 生成其对应的anchors模板
            self.generate_anchors(sizes, aspect_ratios, dtype, device)
            for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)
        ]
        # cell_anchors的形状: List[Tensor]
        # <1> Tensor的形状: (num_anchors, 4), 不同feature map的num_anchors是不同的
        # <2> num_anchors = len(sizes)*len(aspect_ratios)
        self.cell_anchors = cell_anchors

    def num_anchors_per_location(self) -> List[int]:
        """
        返回每一层feature map的每个像素对应的anchors数量
        """
        return [
            len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)
        ]

    def grid_anchors(self, grid_sizes: List[List[int]],
                     strides: List[List[Tensor]]) -> List[Tensor]:
        """
        计算每层feature map对应原始图像上的所有anchors的坐标
    
        参数:
        grid_sizes: List[(h, w)], 每层feature map的h和w
        strides: List[[stride_h, stride_w]], 每层feature map上的1步等于原始图像上的步长(多少像素)

        返回:
        anchors: List[Tensor]
          每个Tensor表示一层feature map对应的所有anchors
          Tensor的形状: (h*w*num_anchors, 4), 不同feature map的h/w/num_anchors是不同的
        """
        anchors = []
        cell_anchors = self.cell_anchors
        assert cell_anchors is not None

        # 遍历每层feature map:
        # strides: List[[stride_h, stride_w]], 每层feature map上的1步等于原始图像上的步长(多少像素)
        # grid_sizes: List[(h, w)], 每层feature map的h和w
        # cell_anchors的形状: List[Tensor], 其中Tensor的形状: (num_anchors, 4)
        for size, stride, base_anchors in zip(grid_sizes, strides,
                                              cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            # shifts_x的形状: (w, )
            shifts_x = torch.arange(
                0, grid_width, dtype=torch.float32,
                device=device) * stride_width
            # shifts_y的形状: (h, )
            shifts_y = torch.arange(
                0, grid_height, dtype=torch.float32,
                device=device) * stride_height

            # shift_y的形状: (h, w)
            # shift_x的形状: (h ,w)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            # shift_y的形状: (h*w, )
            # shift_x的形状: (h*w, )
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            # shifts的形状: (h*w, 4)
            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)

            # shifts_anchor的形状:
            # (h*w, 1, 4) + (1,num_anchors,4) ->
            # (h*w, num_anchors, 4)
            shifts_anchor = shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)
            # (h*w*num_anchors, 4)
            anchors.append(shifts_anchor.reshape(-1, 4))

        return anchors

    def cached_grid_anchors(self, grid_sizes: List[List[int]],
                            strides: List[List[Tensor]]) -> List[Tensor]:
        """
        计算 & 缓存每层feature map对应原始图像上的所有anchors的坐标

        如果grid_sizes和strides固定, 则其对应的anchors也是固定的, 所以可以用这两个
        作为key来cache数据, 避免重复计算

        参数:
        grid_sizes: List[(h, w)], 每层feature map的h和w
        strides: List[[stride_h, stride_w]], 每层feature map上的1步等于原始图像上的步长(多少像素)

        返回:
        anchors: List[Tensor]
          每个Tensor表示一层feature map对应的所有anchors
          Tensor的形状: (h*w*num_anchors, 4), 不同feature map的h/w/num_anchors是不同的
        """
        key = str(grid_sizes) + str(strides)
        # self._cache是字典类型
        if key in self._cache:
            return self._cache[key]
        anchors = self.grid_anchors(grid_sizes, strides)
        self._cache[key] = anchors
        return anchors

    def forward(self, image_list: ImageList,
                feature_maps: List[Tensor]) -> List[Tensor]:
        """
        将一个批量的图片image_list以及其backbone输出的feature_maps作为输入, 输出
        每张图片对应的anchors(是由feature map上的anchors映射过去的)

        参数:
        image_list:
          ImageList: 包含下面两部分信息
          a. ImageList.tensors: padding后的图像数据, 图像具有相同的尺寸: (batch_size, C, H_new, W_new)
          b. ImageList.image_size: list of (H, W) padding前的图像尺寸, 一共batch_size个元素
        features: List[Tensor]
          <1> 每个Tensor表示一层feature map, 有多层feature map, 就有多个Tensor
          <2> Tensor的形状: (batch_size, 1280, h, w) 
              1280是feature map的通道数
              h/w是feature map的高和宽, 不同feature map的h/w是不同的
        
        返回:
        anchors: List[Tensor]
          第一层List是这个batch还有多少张图片, 也就是batch_size
          Tensor的形状: (h1*w1*num_anchors1 + h2*w2*num_anchors2 + ..., 4)
        """
        # 每层feature map的h和w
        # grid_sizes: List[(h, w)]
        grid_sizes = list(
            [feature_map.shape[-2:] for feature_map in feature_maps])

        # 获取输入图像的H和W, 图像的尺寸都是一样的
        # image_size: (H, W)
        image_size = image_list.tensors.shape[-2:]

        # 获取变量类型和设备类型
        dtype, device = feature_maps[0].dtype, feature_maps[0].device

        # 每层feature map上的1步等于原始图像上的步长(多少像素)
        # strides的形状: List[[stride_h, stride_w]]
        strides = [[
            torch.tensor(image_size[0] // g[0],
                         dtype=torch.int64,
                         device=device),
            torch.tensor(image_size[1] // g[1],
                         dtype=torch.int64,
                         device=device)
        ] for g in grid_sizes]

        # 计算每一层feature map的anchors模板(也就是feature map的每个像素如何生成anchors)
        # 数据被保存在了: self.cell_anchors, 其形状为: List[Tensor]
        # <1> Tensor的形状: (num_anchors, 4), 不同feature map的num_anchors是不同的
        # <2> num_anchors = len(sizes)*len(aspect_ratios)
        self.set_cell_anchors(dtype, device)

        # anchors_over_all_feature_maps： List[Tensor]
        # <1> 每个Tensor表示一层feature map对应的所有anchors
        # <2> Tensor的形状: (h*w*num_anchors, 4), 不同feature map的h/w/num_anchors是不同的
        anchors_over_all_feature_maps = self.cached_grid_anchors(
            grid_sizes, strides)

        # anchors: List[List[Tensor]]
        # 1. 第一层List是这个batch有多少张图片, 也就是batch_size
        # 2. 第二层List是每个图片经过CNN输出多少层feature map
        # 3. Tensor的形状: (h*w*num_anchors, 4), 不同feature map的h/w/num_anchors是不同的
        anchors = torch.jit.annotate(List[List[Tensor]], [])
        # 遍历一个batch中的每张图像
        for _, _ in enumerate(image_list.image_sizes):
            anchors_in_image = []
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(anchors_in_image)
        # 将每一张图像的所有feature map的anchors坐标信息拼接在一起
        # anchors: List[Tensor]
        # 1. 第一层List是这个batch还有多少张图片, 也就是batch_size
        # 2. Tensor的形状: (h1*w1*num_anchors1 + h2*w2*num_anchors2 + ..., 4)
        anchors = [
            torch.cat(anchors_per_image) for anchors_per_image in anchors
        ]
        self._cache.clear()
        return anchors


class RPNHead(nn.Module):
    """
    将backbone输出的feature map送入RPNHead, RPNHead会输出feature map上每个像素对应的
    anchors的`分类`和`offset结果`

    1. 3x3滑动窗口提取多层feature map上的每个像素点的所有anchors的特征
    2. 预测每个锚框的分类(2分类)
    3. 预测每个anchor的offset

    针对每层feature map, 输出的高度和宽度不变, 变化的是其通道数

    >>> features = [torch.randn(2, 1280, 32, 32)] # 这个例子只有一层feature map
    >>> head = RPNHead(1280, 5)
    >>> logits, bbox_reg = head(features)
    >>> assert logits[0].shape == (2, 5, 32, 32)       # 第一层feature的logits
    >>> assert bbox_reg[0].shape == (2, 5 * 4, 32, 32) # 第一层feature的bbox_reg
    """
    def __init__(self, in_channels: int, num_anchors: int) -> None:
        """
        参数:
        in_channels: feature map的通道数(也就是经过backbone处理之后的图片的通道数)
        num_anchors: feature map上的每个像素(每个滑动窗口)要预测的anchors数量
        """
        super(RPNHead, self).__init__()
        # 3x3滑动窗口提取feature map上每个像素点的所有anchors的特征
        # 这里的特征也是in_channels维的
        self.conv = nn.Conv2d(in_channels,
                              in_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        # 预测每个锚框的分类(2分类)
        # 注意: 这里的分类目标只是指`前景`或者`背景`
        self.cls_logits = nn.Conv2d(in_channels,
                                    num_anchors,
                                    kernel_size=1,
                                    stride=1)
        # 预测每个anchor的offset
        self.bbox_pred = nn.Conv2d(in_channels,
                                   num_anchors * 4,
                                   kernel_size=1,
                                   stride=1)

        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self,
                features: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """
        RPNHead会输出feature map上每个像素对应的anchors的分类和offset结果

        参数:
        features: List[Tensor]
        <1> 每个Tensor表示一层feature map, 有多层feature map, 就有多个Tensor
        <2> Tensor的形状: (batch_size, 1280, h, w) 
            1280是feature map的通道数
            h/w是feature map的高和宽, 不同feature map的h/w是不同的

        返回: logits, bbox_reg
        logits: List[Tensor]
          <1> 每个Tensor表示一层feature map, 有多层feature map, 就有多个tensor
          <2> Tensor的形状: (batch_size, num_anchors, h, w)
              不同feature map的num_anchors/h/w都是不同的
        bbox_reg: List[Tensor]
          <1> 每个Tensor表示一层feature map, 有多层feature map, 就有多个tensor
          <2> Tensor的形状: (batch_size, num_anchors*4, h, w)
              不同feature map的num_anchors/h/w都是不同的
        """
        logits = []
        bbox_reg = []
        for i, feature in enumerate(features):
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


def permute_and_flatten(layer: Tensor, N: int, A: int, C: int, H: int,
                        W: int) -> Tensor:
    """
    参数:
    layer: 两种情况:
      情况1: (batch_size, num_anchors, h, w), 一层feature map对应的预测概率
      情况2: (batch_size, num_anchors*4, h, w), 一层feature map对应的预测offset
    N: batch_size
    A: num_anchors, 每个位置上的anchor数量
    C: 情况1: classes_num=1
       情况2: 4
    H: h, feature map的高度
    W: w, feature map的宽度

    返回:
    layer: 
      情况1: (batch_size, h*w*num_anchors, 1)
      情况2: (batch_size, h*w*num_anchors, 4)
    """
    # layer的形状:
    # 情况1: (batch_size, num_anchors, h, w) -> (batch_size, num_anchors, 1, h, w)
    # 情况2: (batch_size, num_anchors*4, h, w) -> (batch_size, num_anchors, 4, h, w)
    layer = layer.view(N, -1, C, H, W)
    # layer的形状:
    # 情况1: (batch_size, h, w, num_anchors, 1)
    # 情况2: (batch_size, h, w, num_anchors, 4)
    layer = layer.permute(0, 3, 4, 1, 2)
    # layer的形状:
    # 情况1: (batch_size, h*w*num_anchors, 1)
    # 情况2: (batch_size, h*w*num_anchors, 4)
    layer = layer.reshape(N, -1, C)
    return layer


def concat_box_prediction_layers(
        box_cls: List[Tensor],
        box_regression: List[Tensor]) -> Tuple[Tensor, Tensor]:
    """
    对预测结果目标概率和offset拉平(方便后续处理)

    参数:
    box_cls: List[Tensor] 每层feature map预测的目标概率
      <1> 每个Tensor表示一层feature map, 有多层feature map, 就有多个tensor
      <2> Tensor的形状: (batch_size, num_anchors, h, w)
          不同feature map的num_anchors/h/w都是不同的
    box_regression: List[Tensor] 每层feature map预测的offset
      <1> 每个Tensor表示一层feature map, 有多层feature map, 就有多个tensor
      <2> Tensor的形状: (batch_size, num_anchors*4, h, w)
          不同feature map的num_anchors/h/w都是不同的

    返回: Tuple[box_cls, box_regression]
    box_cls: 
      有一层feature map的情况: (batch_size*h*w*num_anchors, 1)
      有多层feature map的情况: (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 1)
    box_regression: 
      有一层feature map的情况: (batch_size*h*w*num_anchors, 4)
      有多层feature map的情况: (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 4)
    """
    box_cls_flattened = [
    ]  # List[Tensor], Tensor的形状: (batch_size, h*w*num_anchors, 1)
    box_regression_flattened = [
    ]  # List[Tensor], Tensor的形状: (batch_size, h*w*num_anchors, 4)

    # 遍历每层feature map
    # box_cls_per_level的形状: (batch_size, num_anchors, h, w)
    # box_regression_per_level的形状: (batch_size, num_anchors*4, h, w)
    for box_cls_per_level, box_regression_per_level in zip(
            box_cls, box_regression):
        # AxC=num_anchors
        N, AxC, H, W = box_cls_per_level.shape
        # Ax4=num_anchors*4
        Ax4 = box_regression_per_level.shape[1]
        # A=num_anchors
        A = Ax4 // 4
        # C=1
        C = AxC // A

        # box_cls_per_level的形状: (batch_size, h*w*num_anchors, 1)
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H,
                                                W)
        box_cls_flattened.append(box_cls_per_level)

        # box_regression_per_level的形状: (batch_size, h*w*num_anchors, 4)
        box_regression_per_level = permute_and_flatten(
            box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)

    # box_cls的形状:
    # 有一层feature map的情况: (batch_size*h*w*num_anchors, 1)
    # 有多层feature map的情况: (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 1)
    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)
    # 有一层feature map的情况: (batch_size*h*w*num_anchors, 4)
    # 有多层feature map的情况: (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 4)
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression


class RegionProposalNetwork(torch.nn.Module):
    """
    实现RPN(Region Proposal Network)
    """
    __annotations__ = {
        'box_coder': BoxCoder,
        'proposal_matcher': Matcher,
        'fg_bg_sampler': BalancedPositiveNegativeSampler,
        'pre_nms_top_n': Dict[str, int],
        'post_nms_top_n': Dict[str, int],
    }

    def __init__(self,
                 anchor_generator: AnchorsGenerator,
                 head: RPNHead,
                 fg_iou_thresh: float,
                 bg_iou_thresh: float,
                 batch_size_per_image: int,
                 positive_fraction: float,
                 pre_nms_top_n: Dict[str, int],
                 post_nms_top_n: Dict[str, int],
                 nms_thresh: float,
                 score_thresh: float = 0.0) -> None:
        """
        参数:
        anchor_generator: 将`一个批量的图片imageList`以及`其backbone输出的feature_maps`作为输入, 
                          输出`每张图片对应的anchors(是由feature map上的anchors映射过去的)`
        head: 将backbone输出的feature map送入RPNHead, RPNHead会输出feature map上每个像素对应的
              anchors的`分类`和`offset结果`
        fg_iou_thresh: 如果生成的anchors和ground truth的IoU大于这个值, 则标记为正样本. e.g. 0.7
        bg_iou_thresh: 如果生成的anchors和ground truth的IoU小于这个值, 则标记为负样本, e.g. 0.3
        batch_size_per_image: 计算损失时每张图片采用的正负样本的总个数, e.g. 256
        positive_fraction: 正样本占计算损失时所有样本的比例, e.g. 0.5
        pre_nms_top_n: 在NMS处理前保留的proposal数, 应该针对training/testing分别设置
          e.g. dict(training=2000, testing=1000)
        post_nms_top_n: 在NMS处理后保留的proposal数, 应该针对training/testing分别设置
          e.g. dict(training=2000, testing=1000)
        nms_thresh: 进行NMS处理时使用的IoU阈值, 大于这个阈值的会被删除
        score_thresh: 预测的proposals的概率小于这个会被移除
        """
        super(RegionProposalNetwork, self).__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        # 计算anchors与真实bbox的iou
        self.box_similarity = box_iou

        self.proposal_matcher = Matcher(
            fg_iou_thresh,  # 当iou大于fg_iou_thresh(0.7)时视为正样本
            bg_iou_thresh,  # 当iou小于bg_iou_thresh(0.3)时视为负样本
            allow_low_quality_matches=True)

        self.fg_bg_sampler = BalancedPositiveNegativeSampler(
            batch_size_per_image,
            positive_fraction  # 256, 0.5
        )

        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh  # 预测的proposals的概率小于这个会被移除
        self.min_size = 1.  # 宽/高小于min_size的proposals会被移除

    def pre_nms_top_n(self) -> int:
        """
        在NMS处理前保留的proposal数
        """
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']

    def post_nms_top_n(self) -> int:
        """
        在NMS处理后保留的proposal数
        """
        if self.training:
            return self._post_nms_top_n['training']
        return self._post_nms_top_n['testing']

    def assign_targets_to_anchors(
            self, anchors: List[Tensor],
            targets: List[Dict[str,
                               Tensor]]) -> Tuple[List[Tensor], List[Tensor]]:
        """
        根据IoU Matrix来标记anchors & gt-boxes, 并划分为: 正样本, 背景以及废弃的样本

        labels:
        <1> 正样本处标记为1
        <2> 负样本(背景)处标记为0
        <3> 丢弃样本处标记为-1
        
        参数:
        anchors: List[Tensor], 每张图片对应的anchors(是由feature map上的anchors映射过去的)
          1. 第一层List是这个batch有多少张图片, 也就是batch_size
          2. Tensor的形状: (N, 4)
        targets: list of dict, 每个dict包含如下k/v对: 元素个数=batch_size
            boxes    - list of [xmin, ymin, xmax, ymax]
            labels   - 标签列表
            image_id - 图片索引
            area     - 边界框面积
            iscrowd  - 表示目标检测的难易程度: 0表示容易检测; 1表示比较难检测

        返回: labels: List[Tensor], matched_gt_boxes: List[Tensor]
        labels的Tensor形状为: (N, ) - 每个anchors匹配到的类别
          正样本处标记为1
          负样本(背景)处标记为0
          丢弃样本处标记为-1
        matched_gt_boxes的Tensor形状为: (N, 4)
          与anchor匹配的gt box
          注意: 对于负样本/丢弃样本, 我们设置与其匹配的gt box索引为0, 这两类可以通过labels来区分
        """
        labels = []
        matched_gt_boxes = []
        # 遍历每张图像的anchors和targets
        # anchors_per_image的形状: (h1*w1*num_anchors1 + h2*w2*num_anchors2 + ..., 4)
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            # gt_boxes: (num_gt_boxes, 4)
            gt_boxes = targets_per_image["boxes"]
            if gt_boxes.numel() == 0:
                # 没有gt boxes, 则所有的anchors都标记为负类(背景), label都为0
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(
                    anchors_per_image.shape,
                    dtype=torch.float32,
                    device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0], ),
                                               dtype=torch.float32,
                                               device=device)
            else:
                # 计算anchors与真实gt boxes的iou信息
                # set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                # match_quality_matrix的形状:
                # (num_gt_boxes, h1*w1*num_anchors1 + h2*w2*num_anchors2 + ...)
                match_quality_matrix = box_iou(gt_boxes, anchors_per_image)

                # 标记anchors & gt-boxes
                # matched_idxs: (h1*w1*num_anchors1 + h2*w2*num_anchors2 + ..., ) - 每个anchor对应的gt-box索引
                # <1> 正样本, 索引为: [0, num_gt_boxes-1]
                # <2> iou小于low_threshold的matches索引置为: -1
                # <3> iou在[low_threshold, high_threshold]之间的matches索引置为: -2
                matched_idxs = self.proposal_matcher(match_quality_matrix)

                # 注意:
                # 这里使用clamp设置下限0是为了方便取每个anchors对应的gt_boxes信息
                # 负样本和舍弃的样本都是负值, 为了防止越界直接置为0. 因为后面是通过
                # labels_per_image变量来记录正样本的索引, 负样本和舍弃的样本对应的
                # gt_boxes信息并没有什么意义, 我们并不会用到里面的信息
                #
                # matched_gt_boxes_per_image的形状: (h1*w1*num_anchors1 + h2*w2*num_anchors2 + ..., 4)
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(
                    min=0)]

                # 记录所有anchors匹配后的标签
                # <1> 正样本处标记为1
                # <2> 负样本处标记为0
                # <3> 丢弃样本处标记为-1
                # labels_per_image的形状:  (h1*w1*num_anchors1 + h2*w2*num_anchors2 + ..., ) - 内容为False|True
                labels_per_image = matched_idxs >= 0
                # labels_per_image的形状:  (h1*w1*num_anchors1 + h2*w2*num_anchors2 + ..., ) - 内容为0|1
                labels_per_image = labels_per_image.to(dtype=torch.float32)

                # 负样本
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD  # -1
                labels_per_image[bg_indices] = 0.0

                # 丢弃的样本
                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS  # -2
                labels_per_image[inds_to_discard] = -1.0

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes

    def _get_top_n_idx(self, objectness: Tensor,
                       num_anchors_per_level: List[int]) -> Tensor:
        """
        获取每层feature map上预测概率排前pre_nms_top_n的anchors索引值(idx)
        1. 每一层feature map都会选取pre_nms_top_n个bbox

        参数:
        objectness: (batch_size, h1*w1*num_anchors1+h2*w2*num_anchors2+...)
        num_anchors_per_level: List[int], 里面的元素是: num_anchors*h*w
                               不同feature map层的num_anchors/h/w不同
        
        返回:
        idx: (batch_size, pre_nms_top_n1 + pre_nms_top_n2+...)
        """
        r = []  # 记录每层feature map上预测目标概率前pre_nms_top_n的索引信息
        offset = 0
        # 遍历每层feature map上的预测目标概率信息
        for ob in objectness.split(num_anchors_per_level, 1):
            # ob的形状: (batch_size, h*w*num_anchors)
            all_num_anchors = ob.shape[
                1]  # 当前这一层feature map的anchors数=h*w*num_anchors
            pre_nms_top_n = min(self.pre_nms_top_n(), all_num_anchors)

            # top_n_idx的形状: (batch_size, pre_nms_top_n)
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += all_num_anchors
        # (batch_size, pre_nms_top_n1 + pre_nms_top_n2+...)
        return torch.cat(r, dim=1)

    def filter_proposals(
            self, proposals: Tensor, objectness: Tensor,
            image_shapes: List[Tuple[int,
                                     int]], num_anchors_per_level: List[int]
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """
        针对当前的proposals进行过滤(去掉不符合条件的proposals)

        1. 根据预测概率获取每层feature map对应的前post_nms_top_n个proposals(其它的删除)
        2. 裁剪proposals, 将越界的坐标调整到图片边界(img_shape)上
        3. 移除宽/高小于min_size的proposals
        4. 移除小概率proposals(概率小于score_thresh)
        5. 针对每层feature map, 单独进行NMS操作
        6. 返回最终留下来的proposals及其对应的分数(概率值)
        
        参数:
        proposals: (batch_size, h1*w1*num_anchors1+h2*w2*num_anchors2+..., 4) - 预测的boxes
        objectness: (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 1) - 预测的目标概率
        image_shapes: list of (H, W) padding前的图像尺寸, 一共batch_size个元素
        num_anchors_per_level: List[int], 里面的元素是: num_anchors*h*w
                               不同feature map层的num_anchors/h/w不同

        返回: final_boxes, final_scores
        final_boxes: List[Tensor] - 最终保留下来的boxes, 按照分数从高到低降序排列
           List的元素个数=batch_size
           Tensor的形状: (K, 4), K为保留下来的proposals个数, 每个图像的K是不同的
        final_scores: List[Tensor] - 最终保留下来的boxes对应的分数(也就是概率值)
           List的元素个数=batch_size
           Tensor的形状: (K, ), K为保留下来的proposals个数, 每个图像的K是不同的
        """
        # num_images=batch_size
        num_images = proposals.shape[0]
        device = proposals.device

        # do not backprop throught objectness
        objectness = objectness.detach()
        # objectness的形状: (batch_size, h1*w1*num_anchors1+h2*w2*num_anchors2+...)
        objectness = objectness.reshape(num_images, -1)

        # 不同层的feature map, 对应不同的idx: 0, 1, 2..., 为什么要有这个操作?
        # 是因为我们做NMS的时候针对不同层feature map做的, 不同的feature map之间做NMS不会相互影响
        # 所以可以把这个level理解成不同feature map的分类
        # levels: List[Tensor]
        # Tensor的形状为: (num_anchors*h*w, ), 不同feature map层的num_anchors/h/w不同, 内容为idx
        levels = [
            torch.full((n, ), idx, dtype=torch.int64, device=device)
            for idx, n in enumerate(num_anchors_per_level)
        ]
        # levels的形状: (num_anchors1*h1*w1+num_anchors2*h2*w2+..., )
        levels = torch.cat(levels, 0)

        # levels的形状: (1, num_anchors1*h1*w1+num_anchors2*h2*w2+...)
        #           -> (batch_size, h1*w1*num_anchors1+h2*w2*num_anchors2+...)
        levels = levels.reshape(1, -1).expand_as(objectness)

        # 获取每层feature map上预测概率排前pre_nms_top_n的anchors索引值(idx)
        # 1. 每一层feature map都会选取pre_nms_top_n个bbox
        # top_n_idx的形状: (batch_size, pre_nms_top_n1 + pre_nms_top_n2+...)
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)

        # image_range的形状: (batch_size, )
        image_range = torch.arange(num_images, device=device)
        # batch_idx的形状: (batch_size, 1)
        batch_idx = image_range[:, None]

        # 根据每层feature map上预测概率排前pre_nms_top_n的anchors索引值获取相应概率信息
        # objectness的形状: (batch_size, pre_nms_top_n1 + pre_nms_top_n2+...)
        objectness = objectness[batch_idx, top_n_idx]
        # levels的形状: (batch_size, pre_nms_top_n1 + pre_nms_top_n2+...)
        levels = levels[batch_idx, top_n_idx]
        # 预测概率排前pre_nms_top_n的anchors索引值获取相应bbox坐标信息
        # proposals的形状: (batch_size, pre_nms_top_n1 + pre_nms_top_n2+..., 4)
        proposals = proposals[batch_idx, top_n_idx]
        # objectness_prob的形状: (batch_size, pre_nms_top_n1 + pre_nms_top_n2+...)
        objectness_prob = torch.sigmoid(objectness)

        final_boxes = []
        final_scores = []
        # 遍历每张图像的相关预测信息
        # boxes的形状: (pre_nms_top_n1 + pre_nms_top_n2, 4) - proposals
        # scores的形状: (pre_nms_top_n1 + pre_nms_top_n2,)  - 预测的概率
        # lvl的形状: (pre_nms_top_n1 + pre_nms_top_n2,)     - 对应的feature map索引
        # img_shape的形状: (H, W) padding前的图像尺寸         - 原始图片尺寸
        for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob,
                                                 levels, image_shapes):
            # 裁剪boxes, 将越界的坐标调整到图片边界(img_shape)上
            boxes = clip_boxes_to_image(boxes, img_shape)

            # 移除宽/高小于min_size的boxes, 返回保留下来的boxes的索引
            # keep: (K, ), 返回的是保留的K个boxes的索引
            keep = remove_small_boxes(boxes, self.min_size)
            # boxes的形状: (K, 4)
            # scores的形状: (K,)
            # lvl的形状: (K,)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # 移除小概率boxes，参考下面这个链接
            # https://github.com/pytorch/vision/pull/3205
            keep = torch.where(torch.ge(scores, self.score_thresh))[0]
            # boxes的形状: (K, 4)
            # scores的形状: (K,)
            # lvl的形状: (K,)
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # 针对`每一层feature map`/`每个类别`单独使用NMS, 这样`每一层feature map`/`每个类别`
            # 对应的boxes不会相互影响
            # keep: (K, ), 返回的是保留的K个boxes的索引, 按照分数从高到低降序排列
            keep = batched_nms(boxes, scores, lvl, self.nms_thresh)

            keep = keep[:self.post_nms_top_n()]
            # boxes的形状: (K, 4)
            # scores的形状: (K,)
            boxes, scores = boxes[keep], scores[keep]
            final_boxes.append(boxes)
            final_scores.append(scores)

        return final_boxes, final_scores

    def compute_loss(
            self, objectness: Tensor, pred_bbox_deltas: Tensor,
            labels: List[Tensor],
            regression_targets: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        计算RPN损失: 包括分类的损失(前景与背景), 边界框回归的损失
        
        参数:
        objectness: 预测的分类概率 (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 1)
        pred_bbox_deltas: 预测的offset (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 4)
        labels: 真实的标签 List[Tensor] - 为每个anchor分配的标签
          Tensor形状为: (h1*w1*num_anchors1 + h2*w2*num_anchors2 + ..., )
          1, 0, -1分别对应: 正样本, 背景, 废弃的样本
        regression_targets: List[Tensor]  - offset标签
          Tensor的形状: (h1*w1*num_anchors1 + h2*w2*num_anchors2 + ..., 4)
        
        返回: objectness_loss, box_loss
        objectness_loss: 分类的损失(前景与背景) Tensor 标量
        box_loss: 边界框回归的损失 Tensor 标量
        """
        # 对正负样本进行采样, 按照一定比例返回正样本和负样本
        # sampled_pos_inds: List[Tensor] - 正样本的mask
        #   Tensor的形状: (h1*w1*num_anchors1 + h2*w2*num_anchors2 + ..., )
        #   正样本的位置会设置为1, 其它位置设置为0
        # sampled_neg_inds: List[Tensor] - 负样本的mask
        #   Tensor的形状: (h1*w1*num_anchors1 + h2*w2*num_anchors2 + ..., )
        #   负样本的位置会设置为1, 其它位置设置为0
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        # sampled_pos_inds的形状: (num_pos, ) - 正样本的位置的索引
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        # sampled_neg_inds的形状: (num_neg, ) - 负样本的位置的索引
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

        # 将所有正负样本索引拼接在一起
        # sampled_inds的形状: (num_pos+num_pos, )正样本/负样本的位置的索引
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        # objectnessd的形状: (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), )
        objectness = objectness.flatten()
        # labels的形状: (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), )
        labels = torch.cat(labels, dim=0)
        # regression_targets的形状: (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 4)
        regression_targets = torch.cat(regression_targets, dim=0)

        # 计算边界框回归损失(只关心正样本)
        box_loss = smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],  # (num_pos, 4) - 正样本 offset预测值
            regression_targets[sampled_pos_inds],  # (num_pos, 4) - 正样本 offset标签
            beta=1 / 9,
            size_average=False,  # 没有返回均值(我们自己计算均值)
        ) / (sampled_inds.numel())

        # 计算目标预测概率损失
        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds],  # (num_pos+num_pos, )
            labels[sampled_inds])  # (num_pos+num_pos, )

        return objectness_loss, box_loss

    def forward(
        self,
        images: ImageList,
        features: Dict[str, Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Tuple[List[Tensor], Dict[str, Tensor]]:
        """
        步骤:
        1. 将backbone输出的features map送入RPNHead, RPNHead会输出feature map上每个像素对应
           的anchors的`分类`和`offset`
        2. 将images和features map送入AnchorsGenerator, 获得每张图片对应的anchors(是由feature
           map上的anchors映射过去的)坐标信息
        3. 将第1步预测的anchors`分类`和offset拉平(方便后续处理)
        4. 根据预测的anchors的`offset`和`anchors的坐标信息`, 计算其对应的boxes的坐标, 这些boxes作为
           proposals
        5. 针对当前的proposals进行过滤(去掉不符合条件的proposals), 剩下的proposals可以返回
           <1> 根据预测概率获取每层feature map对应的前post_nms_top_n个proposals(其它的删除)
           <2> 裁剪proposals, 将越界的坐标调整到图片边界(img_shape)上
           <3> 移除宽/高小于min_size的proposals
           <4> 移除小概率proposals(概率小于score_thresh)
           <5> 针对每层feature map, 单独进行NMS操作
        6. 如果是训练模式, 还要额外计算Loss(如果是测试模式则不需要计算Loss)
        

        Loss的计算流程:
        1. 根据IoU Matrix来标记anchors & gt-boxes, 并划分为: 正样本, 背景以及废弃的样本 [用来计算分类的损失]
        2. 根据proposals/anchors和其匹配的gt boxes计算offset [可以作为训练时的label使用, 用来计算边界框回归的损失]
        3. 计算RPN损失: 包括分类的损失(前景与背景), 边界框回归的损失

        参数:
        images: 
          ImageList: 包含下面两部分信息
          a. ImageList.tensors: padding后的图像数据, 图像具有相同的尺寸: (batch_size, C, H_new, W_new)
          b. ImageList.image_size: list of (H, W) padding前的图像尺寸, 一共batch_size个元素
        features: dict[str, Tensor], e.g. {'0', Tensor0, '1', Tensor1, ...}
          经过backbone输出的feature map, 我们可以有多层features map特征, 有的backbone输出一层feature map, 
          有的backbone输出多层feature map
          每个Tensor表示一层feature map, Tensor的形状(batch_size, 1280, h, w) 其中h/w是经过backbone处理
          之后的特征图的高和宽
        targets: list of dict, 每个dict包含如下k/v对: 元素个数=batch_size
            boxes    - list of [xmin, ymin, xmax, ymax]
            labels   - 标签列表
            image_id - 图片索引
            area     - 边界框面积
            iscrowd  - 表示目标检测的难易程度: 0表示容易检测; 1表示比较难检测
        
        返回: boxes, losses
        boxes: List[Tensor] - 最终保留下来的boxes
          List的元素个数=batch_size
          Tensor的形状: (K, 4), K为保留下来的proposals个数, 每个图像的K是不同的
        losses: Dict[str, Tensor], 训练模式有losses, 测试模式没有losses
          {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg
          }
        """
        # features: List[Tensor]
        # Tensor的形状(batch_size, 1280, h, w)
        features = list(features.values())

        # feature map上每个像素对应的anchors的分类和offset结果:
        # objectness: List[Tensor]
        #   <1> 每个Tensor表示一层feature map, 有多层feature map, 就有多个tensor
        #   <2> Tensor的形状: (batch_size, num_anchors, h, w)
        #       不同feature map的num_anchors/h/w都是不同的
        # pred_bbox_deltas: List[Tensor]
        #   <1> 每个Tensor表示一层feature map, 有多层feature map, 就有多个tensor
        #   <2> Tensor的形状: (batch_size, num_anchors*4, h, w)
        #       不同feature map的num_anchors/h/w都是不同的
        objectness, pred_bbox_deltas = self.head(features)

        # anchors: List[Tensor], 每张图片对应的anchors(是由feature map上的anchors映射过去的)
        # 1. 第一层List是这个batch有多少张图片, 也就是batch_size
        # 2. Tensor的形状:
        #    如果只有一层feature map: (h*w*num_anchors, 4)
        #    如果有多层feature map: (h1*w1*num_anchors1 + h2*w2*num_anchors2 + ..., 4)
        anchors = self.anchor_generator(images, features)

        # batch_size
        num_images = len(anchors)

        # 计算每层feature map上的对应的anchors总数量
        # num_anchors_per_level_shape_tensors的形状: [(num_anchors, h, w)]
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        # num_anchors_per_level: List[int]
        # 里面的元素是: num_anchors*h*w, 不同feature map层的num_anchors/h/w不同
        num_anchors_per_level = [
            s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors
        ]

        # 将第1步预测的anchors`分类`和offset拉平(方便后续处理)
        # objectness:
        #  有一层feature map的情况: (batch_size*h*w*num_anchors, 1)
        #  有多层feature map的情况: (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 1)
        #
        # pred_bbox_deltas:
        #  有一层feature map的情况: (batch_size*h*w*num_anchors, 4)
        #  有多层feature map的情况: (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 4)
        objectness, pred_bbox_deltas = concat_box_prediction_layers(
            objectness, pred_bbox_deltas)

        # 根据预测的anchors的`offset`和`anchors的坐标信息`, 计算其对应的boxes的坐标,
        # 这些boxes作为proposals, 我们后续会对这些proposals进行过滤
        # 注意: 这里我们进行了detach()操作, 因为Faster RCNN并不通过proposals进行后向传播
        # proposals的形状: (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 1, 4)
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        # proposals的形状: (batch_size, h1*w1*num_anchors1+h2*w2*num_anchors2+..., 4)
        proposals = proposals.view(num_images, -1, 4)

        # 针对当前的proposals进行过滤:
        # boxes: List[Tensor] - 最终保留下来的boxes
        #    List的元素个数=batch_size
        #    Tensor的形状: (K, 4), K为保留下来的proposals个数, 每个图像的K是不同的
        # scores: List[Tensor] - 最终保留下来的boxes对应的分数
        #    List的元素个数=batch_size
        #    Tensor的形状: (K, ), K为保留下来的proposals个数, 每个图像的K是不同的
        boxes, scores = self.filter_proposals(proposals, objectness,
                                              images.image_sizes,
                                              num_anchors_per_level)
        losses = {}
        if self.training:
            assert targets is not None
            # labels: List[Tensor] - 为每个anchor分配的标签
            #   Tensor形状为: (h1*w1*num_anchors1 + h2*w2*num_anchors2 + ..., )
            #   正样本处标记为1
            #   负样本(背景)处标记为0
            #   丢弃样本处标记为-1
            # matched_gt_boxes: List[Tensor] - 与anchor匹配的gt box
            #   Tensor形状为: (h1*w1*num_anchors1 + h2*w2*num_anchors2 + ..., 4)
            #   注意: 对于负样本/丢弃样本, 我们设置与其匹配的gt box索引为0, 这两类可以通过labels来区分
            labels, matched_gt_boxes = self.assign_targets_to_anchors(
                anchors, targets)
            # 根据proposals和其匹配的gt boxes计算offset [可以作为训练时的label使用]
            # regression_targets: List[Tensor]
            #   Tensor的形状: (h1*w1*num_anchors1 + h2*w2*num_anchors2 + ..., 4)
            regression_targets = self.box_coder.encode(matched_gt_boxes,
                                                       anchors)
            # 计算Loss:
            # objectness_loss: 分类的损失(前景与背景) Tensor 标量
            # box_loss: 边界框回归的损失 Tensor 标量
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets)
            # losses: Dict[str, Tensor]
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg
            }
        return boxes, losses
