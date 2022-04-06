from typing import List, Dict
from torch import Tensor
from collections import defaultdict
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import numpy as np
import torch
import torchvision
from mobilenet_v2 import mobilenet_v2, MobileNetV2
from faster_rcnn_model import FasterRCNN
from pascal_voc import load_pascal_voc
from rpn import AnchorsGenerator

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige',
    'Bisque', 'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue',
    'AntiqueWhite', 'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk',
    'Crimson', 'Cyan', 'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki',
    'DarkOrange', 'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise',
    'DarkViolet', 'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick',
    'FloralWhite', 'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold',
    'GoldenRod', 'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory',
    'Khaki', 'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon',
    'LightBlue', 'LightCoral', 'LightCyan', 'LightGoldenRodYellow',
    'LightGray', 'LightGrey', 'LightGreen', 'LightPink', 'LightSalmon',
    'LightSeaGreen', 'LightSkyBlue', 'LightSlateGray', 'LightSlateGrey',
    'LightSteelBlue', 'LightYellow', 'Lime', 'LimeGreen', 'Linen', 'Magenta',
    'MediumAquaMarine', 'MediumOrchid', 'MediumPurple', 'MediumSeaGreen',
    'MediumSlateBlue', 'MediumSpringGreen', 'MediumTurquoise',
    'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin', 'NavajoWhite',
    'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed', 'Orchid',
    'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue',
    'GreenYellow', 'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat',
    'White', 'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def filter_low_thresh(boxes: List, scores: List, classes: List,
                      id_to_class: Dict[int, str], thresh: float,
                      box_to_display_str_map: defaultdict,
                      box_to_color_map: defaultdict) -> None:
    """
    参数:
    boxes: list of [xmin, ymin, xmax, ymax]
    scores: 边界框的分数(从高到低排序过的分数)
    classes: 边界框的class id
    id_to_class: dict
    thresh: score小于thresh的边界框不显示
    box_to_display_str_map: collections.defaultdict(list)
    box_to_color_map: collections.defaultdict(str)
    """
    # 遍历所有的边界框
    for i in range(boxes.shape[0]):
        if scores[i] > thresh:
            box = tuple(boxes[i].tolist())  # numpy -> list -> tuple
            if classes[i] in id_to_class.keys():
                class_name = id_to_class[classes[i]]
            else:
                class_name = 'N/A'
            display_str = str(class_name)
            display_str = '{}: {}%'.format(display_str, int(100 * scores[i]))
            box_to_display_str_map[box].append(display_str)
            box_to_color_map[box] = STANDARD_COLORS[classes[i] %
                                                    len(STANDARD_COLORS)]
        else:
            break  # 网络输出概率已经排序过, 当遇到一个不满足后面的肯定不满足, 不需要继续遍历了


def draw_text(draw: ImageDraw, box_to_display_str_map: defaultdict, box: tuple,
              left: float, right: float, top: float, bottom: float,
              color: str) -> None:
    """
    
    """
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [
        font.getsize(ds)[1] for ds in box_to_display_str_map[box]
    ]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in box_to_display_str_map[box][::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)],
                       fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str,
                  fill='black',
                  font=font)
        text_bottom -= text_height - 2 * margin


def draw_box(image: object,
             boxes: List,
             classes: List,
             scores: List,
             id_to_class: Dict[int, str],
             thresh: float = 0.5,
             line_thickness: int = 8) -> None:
    """
    显示image及其对应的边界框

    >>> from pascal_voc import random_sample, id_to_class
    >>> import matplotlib.pyplot as plt
    >>> import torchvision
    >>> image, target = random_sample()[0]
    >>> image = torchvision.transforms.ToPILImage()(image)
    >>> draw_box(
    >>>     image,
    >>>     target["boxes"].numpy(),
    >>>     target["labels"].numpy(),
    >>>     [1 for i in range(len(target["labels"].numpy()))],  # 所有标签的分数都为1, 都显示
    >>>     id_to_class,
    >>>     thresh=0.5,
    >>>     line_thickness=5)
    >>> plt.imshow(image)
    >>> plt.show()

    参数:
    image: PIL Image
    boxes: list of [xmin, ymin, xmax, ymax]
    classes: 边界框的class id
    scores: 边界框的分数(从高到低排序过的分数)
    id_to_class: dict
    thresh: score小于thresh的边界框不显示
    line_thickness: 边界框的宽度
    """
    box_to_display_str_map = defaultdict(list)
    box_to_color_map = defaultdict(str)

    filter_low_thresh(boxes, scores, classes, id_to_class, thresh,
                      box_to_display_str_map, box_to_color_map)

    # Draw all boxes onto image.
    draw = ImageDraw.Draw(image)
    for box, color in box_to_color_map.items():
        xmin, ymin, xmax, ymax = box
        (left, right, top, bottom) = (xmin * 1, xmax * 1, ymin * 1, ymax * 1)
        draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
                   (left, top)],
                  width=line_thickness,
                  fill=color)
        draw_text(draw, box_to_display_str_map, box, left, right, top, bottom,
                  color)


def create_backbone(pretrained: bool = True,
                    num_classes: int = 1000,
                    alpha: float = 1.0,
                    round_nearest: float = 8) -> MobileNetV2:
    """
    创建特征提取的backbone

    其实backbone的类别(num_classes)是没有意义的, 因为我们只关心特征提取的部分

    >>> backbone = create_backbone()
    >>> x = torch.randn((2, 3, 1024, 1024))
    >>> backbone(x).shape
        torch.Size([2, 1280, 32, 32])
    """
    backbone = mobilenet_v2(pretrained, num_classes, alpha,
                            round_nearest).features
    backbone.out_channels = 1280  # 增加这个属性, 后续在创建`RPNHead`需要用到这个属性
    return backbone


def create_model(num_classes: int = 21) -> FasterRCNN:
    """
    创建Faster RCNN Model
    """
    backbone = create_backbone(num_classes=num_classes)

    anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512), ),
                                        aspect_ratios=((0.5, 1.0, 2.0), ))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],  # 在哪些特征层上进行roi pooling
        output_size=[7, 7],  # roi_pooling输出特征矩阵尺寸
        sampling_ratio=2)  # 采样率

    model = FasterRCNN(backbone=backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    return model


def forward(batch_size=2):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))
    train_iter, _ = load_pascal_voc(batch_size)
    # background + 20 classes
    model = create_model(num_classes=21)
    model.to(device)
    model.train()
    for i, (images, targets) in enumerate(train_iter):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict, _ = model(images, targets)  # FasterRCNN
        losses = sum(loss for loss in loss_dict.values())
        print(f'step:{i} losses:{losses}')
        break


forward(4)