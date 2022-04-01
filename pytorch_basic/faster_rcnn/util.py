from typing import List, Dict, Tuple
from torch import Tensor
from collections import defaultdict
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import numpy as np
import torch

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


class ImageList(object):
    def __init__(self, tensors: Tensor, image_sizes: List[Tuple[int,
                                                                int]]) -> None:
        """
        图像List, 包含`padding后的图像数据`以及`padding前的图像尺寸`

        参数:
        tensors的形状: (batch_size, C, H_new, W_new) padding后的图像数据
        image_sizes: list of (w, h) padding前的图像尺寸
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device: torch.device) -> object:
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)


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