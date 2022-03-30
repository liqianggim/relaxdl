import os
import random
import hashlib
from tkinter import Y
import requests
import zipfile
import tarfile
import json
from PIL import Image
from lxml import etree

import torch
from torch.utils.data import Dataset
from torchaudio import transforms
import torchvision
"""
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

PASCAL VOC(Visual Object Classes)挑战赛是一个世界级的计算机视觉挑战赛,
PASCAL全程(Pattern Analysis, Statical Modeling and Computational Learning), 是一个由欧盟资助的网络组织,
PASCAL VOC挑战赛主要分为如下几类:
1. 目标分类(Object Classification)
2. 目标检测(Object Detection)
3. 目标分割(Object Segmentation)
4. 动作识别(ction Classification)

VOCdevkit/                        - 根目录
    VOC2012/                      - 不同年份的数据集
        Annotations/              - 所有图片的标注信息(XML文件), 与JPEGImages中的图片一一对应
        ImageSets/                - 不容任务的信息
            Action/               - 人的行为动作图像信息
            Layout/               - 人的各个部位图像信息
            Main/                 - 目标检测分类图像信息
                ...
                boat_train.txt    - 针对boat这个类别的训练集
                boat_val.txt      - 针对boat这个类别的验证集
                boat_trainval.txt - 针对boat这个类别的训练集+验证集
                train.txt         - 训练集: 5717
                val.txt           - 验证集: 5823
                trainval.txt      - 训练集+验证集: 11540
                ...
            Segmentation/         - 目标分割图像信息
        JPEGImages/               - 存放源图片
        SegmentationClass/        - 目标分割PNG图(基于类别)
        SegmentationObject/       - 目标分割PNG图(基于目标)
"""
"""
VOC2012/Annotations/2007_002895.xml

<annotation>
	<folder>VOC2012</folder>
	<filename>2007_002895.jpg</filename>
	<source>
		<database>The VOC2007 Database</database>
		<annotation>PASCAL VOC2007</annotation>
		<image>flickr</image>
	</source>
	<size>
		<width>500</width>
		<height>375</height>
		<depth>3</depth>
	</size>
	<segmented>1</segmented> - 表示图像是否进行分割: 没有被分割用0表示, 分割过用1表示
	<object>
		<name>person</name>
		<pose>Rear</pose>
		<truncated>0</truncated> - 表示目标有没有被截断: 0表示没有被截断; 1表示被截断(也就是目标不是完整的)
		<difficult>0</difficult> - 表示目标检测的难易程度: 0表示容易检测; 1表示比较难检测
		<bndbox>
			<xmin>388</xmin>
			<ymin>194</ymin>
			<xmax>419</xmax>
			<ymax>339</ymax>
		</bndbox>
	</object>
	<object>
		<name>person</name>
		<pose>Rear</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>415</xmin>
			<ymin>192</ymin>
			<xmax>447</xmax>
			<ymax>338</ymax>
		</bndbox>
	</object>
</annotation>
"""
"""
VOC2012/ImageSets/Main/train.txt - 每一行表示一个样本
...
2008_000008
2008_000015
2008_000019
2008_000023
...
"""
"""
VOC2012/ImageSets/Main/boat_train.txt
...
2008_007156  1    - 表示正样本
2008_007161 -1    - 表示负样本
2008_007165 -1
2008_007168 -1
2008_007169 -1
2008_007179  0    - 表示很难检测
...
"""


def download(cache_dir='../data'):
    """
    下载数据

    """
    sha1_hash = '225f2d60c6c00936e8725953ad026b9491f22a33'
    url = 'https://foxrelax.oss-cn-hangzhou.aliyuncs.com/ml/VOCdevkit.zip'
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
    # e.g. ../data/VOCdevkit.zip
    return fname


def download_extract(cache_dir='../data'):
    """
    下载数据 & 解压
    """
    # 下载数据集
    fname = download(cache_dir)

    # 解压
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files can be extracted.'
    fp.extractall(base_dir)
    # e.g. ../data/VOCdevkit
    return data_dir


class_to_idx = {
    "aeroplane": 1,
    "bicycle": 2,
    "bird": 3,
    "boat": 4,
    "bottle": 5,
    "bus": 6,
    "car": 7,
    "cat": 8,
    "chair": 9,
    "cow": 10,
    "diningtable": 11,
    "dog": 12,
    "horse": 13,
    "motorbike": 14,
    "person": 15,
    "pottedplant": 16,
    "sheep": 17,
    "sofa": 18,
    "train": 19,
    "tvmonitor": 20
}

id_to_class = {v: k for k, v in class_to_idx.items()}


class VOCDataSet(Dataset):
    """
    读取解析PASCAL VOC2012数据集
    """
    def __init__(self,
                 voc_root=None,
                 year="2012",
                 transforms=None,
                 txt_name="train.txt"):
        """
        参数:
        voc_root: 存放VOCdevkit的路径
        year: 2012
        transforms: 自定义transforms
        txt_name: e.g. train.txt | val.txt
        """
        # VOCdevkit/VOC2012
        self.root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")
        # VOCdevkit/VOC2012/JPEGImages
        self.img_root = os.path.join(self.root, "JPEGImages")
        # VOCdevkit/VOC2012/Annotations
        self.annotations_root = os.path.join(self.root, "Annotations")

        # VOCdevkit/VOC2012/ImageSets/Main/train.txt
        # VOCdevkit/VOC2012/ImageSets/Main/val.txt
        # 每行一个样本
        # ...
        # 2008_000008
        # 2008_000015
        # 2008_000019
        # 2008_000023
        # ...
        txt_path = os.path.join(self.root, "ImageSets", "Main", txt_name)
        assert os.path.exists(txt_path), "not found {} file.".format(txt_name)

        with open(txt_path) as read:
            # [...
            #  VOCdevkit/VOC2012/Annotations/2008_000008.xml
            #  VOCdevkit/VOC2012/Annotations/2008_000015.xml
            #  VOCdevkit/VOC2012/Annotations/2008_000019.xml
            #  VOCdevkit/VOC2012/Annotations/2008_000023.xml
            # ...]
            self.xml_list = [
                os.path.join(self.annotations_root,
                             line.strip() + ".xml")
                for line in read.readlines() if len(line.strip()) > 0
            ]

        # check file
        assert len(
            self.xml_list
        ) > 0, "in '{}' file does not find any information.".format(txt_path)
        for xml_path in self.xml_list:
            assert os.path.exists(xml_path), "not found '{}' file.".format(
                xml_path)

        self.class_to_idx = class_to_idx
        self.transforms = transforms

    def __len__(self):
        """
        样本数量

        Annotations/*.xml文件的个数
        """
        return len(self.xml_list)

    def __getitem__(self, idx):
        """
        没有使用transforms的例子:
        >>> train_dataset = VOCDataSet(voc_root, txt_name='train.txt')
        >>> image, target = train_dataset[0]
        >>> type(image)
        <class 'PIL.JpegImagePlugin.JpegImageFile'>
        >>> target
        {
            'boxes': tensor([[ 53.,  87., 471., 420.],
                             [158.,  44., 289., 167.]]), 
            'labels': tensor([13, 15]), 
            'image_id': tensor([0]), 
            'area': tensor([139194.,  16113.]), 
            'iscrowd': tensor([0, 0])
        }

        原始XML文件格式如下:
        <annotation>
            <folder>VOC2012</folder>
            <filename>2007_002895.jpg</filename>
            <source>
                <database>The VOC2007 Database</database>
                <annotation>PASCAL VOC2007</annotation>
                <image>flickr</image>
            </source>
            <size>
                <width>500</width>
                <height>375</height>
                <depth>3</depth>
            </size>
            <segmented>1</segmented> - 表示图像是否进行分割: 没有被分割用0表示, 分割过用1表示
            <object>
                <name>person</name>
                <pose>Rear</pose>
                <truncated>0</truncated> - 表示目标有没有被截断: 0表示没有被截断; 1表示被截断(也就是目标不是完整的)
                <difficult>0</difficult> - 表示目标检测的难易程度: 0表示容易检测; 1表示比较难检测
                <bndbox>
                    <xmin>388</xmin>
                    <ymin>194</ymin>
                    <xmax>419</xmax>
                    <ymax>339</ymax>
                </bndbox>
            </object>
            <object>
                <name>person</name>
                <pose>Rear</pose>
                <truncated>0</truncated>
                <difficult>0</difficult>
                <bndbox>
                    <xmin>415</xmin>
                    <ymin>192</ymin>
                    <xmax>447</xmax>
                    <ymax>338</ymax>
                </bndbox>
            </object>
        </annotation>

        参数:
        idx: int文件索引

        返回: (image, target)
        image: PIL.JpegImagePlugin.JpegImageFile | Tensor
        target: dict
          boxes    - list of [xmin, ymin, xmax, ymax]
          labels   - 标签列表
          image_id - 图片索引
          area     - 边界框面积
          iscrowd  - 表示目标检测的难易程度: 0表示容易检测; 1表示比较难检测

        """
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        img_path = os.path.join(self.img_root, data["filename"])
        image = Image.open(img_path)
        if image.format != "JPEG":
            raise ValueError("Image '{}' format not JPEG".format(img_path))

        boxes = []
        labels = []
        iscrowd = []
        assert "object" in data, "{} lack of object information.".format(
            xml_path)
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])

            # 进一步检查数据, 有的标注信息中可能有w或h为0的情况, 这样的数据会导致计算回归loss为nan
            if xmax <= xmin or ymax <= ymin:
                print(
                    "Warning: in '{}' xml, there are some bbox w/h <=0".format(
                        xml_path))
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[obj["name"]])
            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes  # [xmin, ymin, xmax, ymax]
        target["labels"] = labels  # 标签
        target["image_id"] = image_id  # 图片索引
        target["area"] = area  # 边界框面积
        target["iscrowd"] = iscrowd  # 表示目标检测的难易程度: 0表示容易检测; 1表示比较难检测

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def get_height_and_width(self, idx):
        """
        获取图片的高和宽
        
        >>> train_dataset = VOCDataSet(voc_root, txt_name='train.txt')
        >>> height, width = train_dataset.get_height_and_width(0)
        (442, 500)

        参数:
        idx: int文件索引
        返回: (height, width)
        height: int
        width: int
        """
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

    def parse_xml_to_dict(self, xml):
        """
        将xml文件解析成dict形式(参考tensorflow的recursive_parse_xml_to_dict)

        返回结果格式:
        {
            'annotation': {
                'folder': 'VOC2012', 
                'filename': '2008_000008.jpg', 
                'source': {
                    'database': 'The VOC2008 Database', 
                    'annotation': 'PASCAL VOC2008', 
                    'image': 'flickr'
                }, 
                'size': {'width': '500', 'height': '442', 'depth': '3'}, 
                'segmented': '0', 
                'object': [
                    {
                        'name': 'horse', 
                        'pose': 'Left', 
                        'truncated': '0', 
                        'occluded': '1', 
                        'bndbox': {'xmin': '53', 'ymin': '87', 'xmax': '471', 'ymax': '420'}, 
                        'difficult': '0'
                    }, 
                    {
                        'name': 'person', 
                        'pose': 'Unspecified', 
                        'truncated': '1', 
                        'occluded': '0', 
                        'bndbox': {'xmin': '158', 'ymin': '44', 'xmax': '289', 'ymax': '167'}, 
                        'difficult': '0'
                    }
                ]
            }
        }
        参数: 
        xml: lxml.etree
        
        返回:
        dict格式的xml文件
        """

        if len(xml) == 0:  # 遍历到底层, 直接返回tag对应的信息
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能有多个, 所以需要放入列表里
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    def coco_index(self, idx):
        """
        该方法是专门为pycocotools统计标签信息准备, 不对图像和标签作任何处理.
        由于不用去读取图片, 可大幅缩减统计时间

        >>> train_dataset = VOCDataSet(voc_root, txt_name='train.txt')
        >>> (data_height, data_width), target = train_dataset.coco_index(0)
        >>> (data_height, data_width)
        (442, 500)
        >>> target
        {
            'boxes': tensor([[ 53.,  87., 471., 420.],
                             [158.,  44., 289., 167.]]), 
            'labels': tensor([13, 15]), 
            'image_id': tensor([0]), 
            'area': tensor([139194.,  16113.]), 
            'iscrowd': tensor([0, 0])
        }

        参数:
        idx: int文件索引

        返回: ((data_height, data_width), target)
        data_height: int
        data_width: int
        target: dict
          boxes    - list of [xmin, ymin, xmax, ymax]
          labels   - 标签列表
          image_id - 图片索引
          area     - 边界框面积
          iscrowd  - 表示目标检测的难易程度: 0表示容易检测; 1表示比较难检测
        """
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        # img_path = os.path.join(self.img_root, data["filename"])
        # image = Image.open(img_path)
        # if image.format != "JPEG":
        #     raise ValueError("Image format not JPEG")
        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[obj["name"]])
            iscrowd.append(int(obj["difficult"]))

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return (data_height, data_width), target

    @staticmethod
    def collate_fn(batch):
        """
        注意:
        一个批量的图片的尺寸是不一样的, 我们需要自己处理一下batch,
        否则直接返回会报错
        """
        return tuple(zip(*batch))


class Compose(object):
    """
    组合多个transform
    函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    """
    将PIL图像转为Tensor
    """
    def __call__(self, image, target):
        image = torchvision.transforms.functional.to_tensor(image)
        return image, target


class RandomHorizontalFlip(object):
    """
    随机水平翻转图像以及bboxes

    注意:
    翻转图像的时候bboxes要一起变化
    """
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            _, width = image.shape[-2:]
            image = image.flip(-1)  # 水平翻转图片
            bbox = target["boxes"]
            # bbox: xmin, ymin, xmax, ymax
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # 翻转对应bbox坐标信息
            target["boxes"] = bbox
        return image, target


def load_pascal_voc(batch_size=8, root='../data'):
    """
    加载PASCAL VOC2012数据集

    批量大小为2, 返回的images,targets都是一个tuple. 第1张图片有一个边界框; 第2张图片有2个边界框
    >>> train_iter, val_iter = load_pascal_voc(batch_size=2, root='../data')
    >>> for images, targets in train_iter:
    >>>     print(images[0].shape)
    >>>     print(targets[0])
    >>>     print(images[1].shape)
    >>>     print(targets[1])
    >>>     break
    torch.Size([3, 500, 375])
    {
       'boxes': tensor([[  0.,  60., 374., 500.]]),
       'labels': tensor([8]),
       'image_id': tensor([3482]),
       'area': tensor([164560.]),
       'iscrowd': tensor([0])
    }
    torch.Size([3, 375, 500])
    {
       'boxes': tensor([[107., 216., 340., 282.],
                        [258.,  20., 422., 263.]]),
       'labels': tensor([ 5, 12]),
       'image_id': tensor([2426]),
       'area': tensor([15378., 39852.]),
       'iscrowd': tensor([0, 0])
    }

    """
    voc_vocdevkit = download_extract()
    voc_root = os.path.dirname(voc_vocdevkit)

    data_transform = {
        'train': Compose([ToTensor(), RandomHorizontalFlip(0.5)]),
        'val': Compose([ToTensor()])
    }
    train_dataset = VOCDataSet(voc_root,
                               transforms=data_transform['train'],
                               txt_name='train.txt')
    train_iter = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn)
    val_dataset = VOCDataSet(voc_root,
                             transforms=data_transform['val'],
                             txt_name='val.txt')
    val_iter = torch.utils.data.DataLoader(val_dataset,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           pin_memory=True,
                                           collate_fn=val_dataset.collate_fn)
    return train_iter, val_iter


def random_sample(n=1):
    """
    从验证集中随机采样n个样本

    >>> samples = random_sample()
    >>> samples[0][0].shape
        torch.Size([3, 333, 500])

    返回: list of (image, target)
    image: Tensor
    target: dict
        boxes    - list of [xmin, ymin, xmax, ymax]
        labels   - 标签列表
        image_id - 图片索引
        area     - 边界框面积
        iscrowd  - 表示目标检测的难易程度: 0表示容易检测; 1表示比较难检测
    """
    voc_vocdevkit = download_extract()
    voc_root = os.path.dirname(voc_vocdevkit)

    data_transform = Compose([ToTensor()])
    val_dataset = VOCDataSet(voc_root,
                             transforms=data_transform,
                             txt_name='val.txt')
    samples = []
    for idx in random.sample(range(0, len(val_dataset)), k=5):
        img, target = val_dataset[idx]
        samples.append((img, target))
    return samples