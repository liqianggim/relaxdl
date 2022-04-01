import math
from typing import List, Tuple, Dict, Optional

import torch
from torch import nn, Tensor

from util import ImageList


def _resize_image(image: Tensor, self_min_size: float,
                  self_max_size: float) -> Tensor:
    """
    缩放一张图片, 确保缩放后的图片尺寸在[self_min_size, self_max_size]范围内

    参数:
    image的形状: (C, H, W)
    self_min_size: float, 图像的最小边长
    self_max_size: float, 图像的最大边长

    返回:
    image的形状: (C, H_new, W_new)
    """
    # 图像的原始尺寸: (H, W)
    im_shape = torch.tensor(image.shape[-2:])
    min_size = float(torch.min(im_shape))  # 获取高宽中的最小值
    max_size = float(torch.max(im_shape))  # 获取高宽中的最大值

    # 计算一个合理的scale_factor, 确保缩放后的高和宽都在[self_min_size, self_max_size]范围内
    scale_factor = self_min_size / min_size  # 根据指定最小边长和图片最小边长计算缩放比例
    if max_size * scale_factor > self_max_size:
        scale_factor = self_max_size / max_size  # 将缩放比例设为指定最大边长和图片最大边长之比

    # interpolate利用插值的方法缩放图片
    # image的形状:(C, H, W) -> (1, C, H, W)
    image = torch.nn.functional.interpolate(image[None],
                                            scale_factor=scale_factor,
                                            mode="bilinear",
                                            recompute_scale_factor=True,
                                            align_corners=False)[0]

    # (C, H_new, W_new)
    return image


class GeneralizedRCNNTransform(nn.Module):
    """
    将images和target送入GeneralizedRCNN之前做transform
    1. 遍历所有的图片
        <1> 对图片进行标准化处理: self.normalize()
        <2> 将图片(及其对应的bboxes)缩放到指定的大小范围内: self.resize()
    2. 将图片处理成形状相同的批量: self.batch_images()
    3. 返回处理后的数据: ImageList, targets
        <1> ImageList: 包含`padding后的图像数据`以及`padding前的图像尺寸`
        <2> targets: 处理后的targets, 因为图片进行了resize, 所以其对应的bbox也做相应的缩放

    Exmaple:
    >>> from pascal_voc import load_pascal_voc
    >>> min_size = 800
    >>> max_size = 1333
    >>> image_mean = [0.485, 0.456, 0.406]
    >>> image_std = [0.229, 0.224, 0.225]
    >>> transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
    >>> train_iter, test_iter = load_pascal_voc(8)
    >>> for images, targets in train_iter:
    >>>     images = [image for image in images]
    >>>     targets = [{k: v for k, v in t.items()} for t in targets]
    >>>     imageList, targets = transform(images, targets)
    >>>     print(imageList.tensors.shape)
    >>>     print(imageList.image_sizes)
    >>>     print(targets[0])
    >>>     break
        torch.Size([8, 3, 1216, 1216])
        [(800, 1066), (800, 1066), (800, 1201), (800, 1126),
         (800, 904), (1201, 800), (1201, 800), (800, 1066)]
         {
          'boxes': tensor([[  29.8480,   46.9333, 1055.3400,  782.9333]]),
          'labels': tensor([8]),
          'image_id': tensor([3984]),
          'area': tensor([165945.]),
          'iscrowd': tensor([0])
        }
    """
    def __init__(self, min_size: int, max_size: int, image_mean: List[float],
                 image_std: List[float]) -> None:
        """
        缩放后的图片尺寸会在: [min_size, max_size]范围内

        参数:
        min_size: 图像的最小边长
        max_size: 图像的最大边长
        image_mean: 图像在标准化处理中的均值 e.g. [0.485, 0.456, 0.406]
        image_std: 图像在标准化处理中的方差 e.g. [0.229, 0.224, 0.225]
        """
        super(GeneralizedRCNNTransform, self).__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size, )
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std

    def normalize(self, image: Tensor) -> Tensor:
        """
        标准化处理

        参数:
        image的形状: (C, H, W)

        返回:
        output的形状: (C, H, W)
        """
        dtype, device = image.dtype, image.device
        # mean的形状: (3,)
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        # std的形状: (3,)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        # mean的形状: (3, ) -> (3, 1, 1)
        # std的形状: (3, ) -> (3, 1, 1)
        # 返回值的形状: (C, H, W)
        return (image - mean[:, None, None]) / std[:, None, None]

    def torch_choice(self, k: List[int]) -> int:
        """
        Implements `random.choice` via torch ops so it can be compiled with
        TorchScript. Remove if https://github.com/pytorch/pytorch/issues/25803
        is fixed.
        """
        index = int(torch.empty(1).uniform_(0., float(len(k))).item())
        return k[index]

    def resize(
        self, image: Tensor, target: Optional[Dict[str, Tensor]]
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        """
        处理一张图片
        将图片(及其对应的bboxes)缩放到指定的大小范围内

        参数:
        image的形状: (C, H, W)
        target: dict, 会更新里面的target['boxes']
            boxes    - list of [xmin, ymin, xmax, ymax]
            labels   - 标签列表
            image_id - 图片索引
            area     - 边界框面积
            iscrowd  - 表示目标检测的难易程度: 0表示容易检测; 1表示比较难检测
        
        返回:
        image的形状: (C, H_new, W_new)
        target: dict
            boxes    - list of [xmin, ymin, xmax, ymax]
            labels   - 标签列表
            image_id - 图片索引
            area     - 边界框面积
            iscrowd  - 表示目标检测的难易程度: 0表示容易检测; 1表示比较难检测
        """
        # 图片的原始尺寸
        h, w = image.shape[-2:]

        # 输入图片的最小边长
        if self.training:
            min_size = float(self.torch_choice(self.min_size))
        else:
            min_size = float(self.min_size[-1])

        # image的形状: (C, H, W) -> (C, H_new, W_new)
        image = _resize_image(image, min_size, float(self.max_size))

        if target is None:
            return image, target

        # bbox的形状: (num_bbox, 4)
        bbox = target["boxes"]
        # 根据图像的缩放比例来缩放bbox
        bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox

        return image, target

    def max_by_axis(self, the_list: (List[List[int]])) -> List[int]:
        """
        找到最大的: [C, H, W]

        参数:
        the_list: list of [C,H,W]

        返回:
        outputs: [C, H, W]
        """
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                # 遍历C, H, W
                maxes[index] = max(maxes[index], item)
        return maxes

    def batch_images(self,
                     images: List[Tensor],
                     size_divisible: int = 32) -> Tensor:
        """
        1. 找到这个批量图片的: 最大的高度H和最大宽度W
        2. 创建一个形状为: batched_imgs.shape = (batch_size, C, H, W)全为0填充的
           4-D Tensor
        3. 将输入images中的每张图片复制到新的batched_imgs的每张图片中, 对齐左上角, 保证bboxes的
           坐标不变, 这样保证输入到网络中一个batch的每张图片的shape相同
        4. 返回这个批量的图片batched_imgs

        参数:
        images: list of Tensor, 其中Tensor.shape = (C, H, W)
        size_divisible: 将图像高和宽调整到该数的整数倍

        返回:
        images: (batch_size, C, H_new, W_new)
        """

        # 计算这一批images中最大的: [C, H, W]
        max_size = self.max_by_axis([list(img.shape) for img in images])

        stride = float(size_divisible)
        # 将height向上调整到stride的整数倍
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        # 将width向上调整到stride的整数倍
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        # [batch_size, C, H, W]
        batch_shape = [len(images)] + max_size

        # 创建shape为batch_shape且值全部为0的tensor
        batched_imgs = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_imgs):
            # 将输入images中的每张图片复制到新的batched_imgs的每张图片中, 对齐左上角,
            # 保证bboxes的坐标不变, 这样保证输入到网络中一个batch的每张图片的shape相同
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)

        return batched_imgs

    def postprocess(
        self, result: List[Dict[str, Tensor]], image_shapes: List[Tuple[int,
                                                                        int]],
        original_image_sizes: List[Tuple[int,
                                         int]]) -> List[Dict[str, Tensor]]:
        """
        对网络的预测结果进行后处理(将bboxes还原到原图像尺度上)

        参数:
        result: list of dict, 每个dict包含如下k/v对:
            boxes    - list of [xmin, ymin, xmax, ymax]
            labels   - 标签列表
            image_id - 图片索引
            area     - 边界框面积
            iscrowd  - 表示目标检测的难易程度: 0表示容易检测; 1表示比较难检测
        image_shapes: list of (w, h)
        original_image_sizes: list of (w, h)
        """
        if self.training:
            return result
        # 遍历每张图片的预测信息, 将boxes信息还原回原尺度
        for i, (pred, im_s, o_im_s) in enumerate(
                zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
        return result

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        _indent = '\n    '
        format_string += "{0}Normalize(mean={1}, std={2})".format(
            _indent, self.image_mean, self.image_std)
        format_string += "{0}Resize(min_size={1}, max_size={2}, mode='bilinear')".format(
            _indent, self.min_size, self.max_size)
        format_string += '\n)'
        return format_string

    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Tuple[ImageList, Optional[List[Dict[str, Tensor]]]]:
        """
        将一个批量的图片处理成可以直接送入网络训练的格式

        1. 遍历所有的图片
           <1> 对图片进行标准化处理: self.normalize()
           <2> 将图片(及其对应的bboxes)缩放到指定的大小范围内: self.resize()
        2. 将图片处理成形状相同的批量: self.batch_images()
        3. 返回处理后的数据: ImageList, targets
           <1> ImageList: 包含`padding后的图像数据`以及`padding前的图像尺寸`
           <2> targets: 处理后的targets, 因为图片进行了resize, 所以其对应的bbox也做相应的缩放

        参数:
        images: list of Tensor, 其中Tensor的形状为: (C, H, W)
        targets: list of dict, 每个dict包含如下k/v对:
            boxes    - list of [xmin, ymin, xmax, ymax]
            labels   - 标签列表
            image_id - 图片索引
            area     - 边界框面积
            iscrowd  - 表示目标检测的难易程度: 0表示容易检测; 1表示比较难检测

        返回:
        images: ImageList, 包含`padding后的图像数据`以及`padding前的图像尺寸`
        targets: list of dict, 每个dict包含如下k/v对:
            boxes    - list of [xmin, ymin, xmax, ymax]
            labels   - 标签列表
            image_id - 图片索引
            area     - 边界框面积
            iscrowd  - 表示目标检测的难易程度: 0表示容易检测; 1表示比较难检测
        """
        images = [img for img in images]

        # <1> 对图片进行标准化处理: self.normalize()
        # <2> 将图片(及其对应的bboxes)缩放到指定的大小范围内: self.resize()
        # 会同步更新targets中boxes的信息
        for i in range(len(images)):
            # 获取图片及其对应的target:
            # image的形状: (C,H,W)
            image = images[i]
            # target_index: dict
            target_index = targets[i] if targets is not None else None

            if image.dim() != 3:
                raise ValueError(
                    "images is expected to be a list of 3d tensors "
                    "of shape [C, H, W], got {}".format(image.shape))
            image = self.normalize(image)  # 对图像进行标准化处理
            image, target_index = self.resize(
                image, target_index)  # 对图像和对应的bboxes缩放到指定范围
            images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index

        # 记录resize后(padding前)的图像尺寸: [[H, W]]
        image_sizes = [img.shape[-2:] for img in images]
        # images的形状: (batch_size, C, H_new, W_new)
        images = self.batch_images(images)  # 将images打包成一个batch
        image_sizes_list = torch.jit.annotate(List[Tuple[int, int]], [])

        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(images, image_sizes_list)
        return image_list, targets


def resize_boxes(boxes: Tensor, original_size: Tuple[int, int],
                 new_size: Tuple[int, int]) -> Tensor:
    """
    将boxes参数根据图像的缩放情况进行相应缩放
    
    参数:
    boxes的形状: (num_boxe, 4)
    original_size: (H, W), 图像的原始H,W
    new_size: (H_new, W_new), 图像缩放后的H,W

    返回:
    boxes的形状: (num_bbox, 4)
    """
    # 图片缩放后的尺寸/图片原始尺寸
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratios_height, ratios_width = ratios
    # xmin的形状: (num_bbox,)
    # xmax的形状: (num_bbox,)
    # ymin的形状: (num_bbox,)
    # ymax的形状: (num_bbox,)
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratios_width
    xmax = xmax * ratios_width
    ymin = ymin * ratios_height
    ymax = ymax * ratios_height
    # (num_bbox, 4)
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)