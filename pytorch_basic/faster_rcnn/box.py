import torch
from typing import Tuple
from torch import Tensor


def nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    """
    实现NMS
    
    参数:
    boxes: (N, 4) - proposals
    scores: (N,) - 预测的概率(用来排序的)
    iou_threshold: float, IoU>iou_threshold的boxes会被删除

    返回:
    Keep: (K,) 返回的是保留的boxes的索引, 按照分数从高到低降序排列
    """
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)


def batched_nms(boxes: Tensor, scores: Tensor, idxs: Tensor,
                iou_threshold: float) -> Tensor:
    """
    针对每一层feature map/每个类别单独使用NMS, 这样每一层feature map/每个类别
    对应的boxes不会相互影响

    参数:
    boxes: (N, 4) - proposals
    scores: (N,) - 预测的概率(用来排序的)
    idxs: (N, ) - 对应的feature map索引, 不同层的boxes不会相互影响
    iou_threshold: float, IoU>iou_threshold的boxes会被删除

    返回:
    Keep: (K,) 返回的是保留的boxes的索引, 按照分数从高到低降序排列
    """
    if boxes.numel() == 0:
        return torch.empty((0, ), dtype=torch.int64, device=boxes.device)

    # 获取所有boxes中最大的坐标值: (xmin, ymin, xmax, ymax）
    max_coordinate = boxes.max()

    # 为每一个类别/每一层生成一个很大的偏移量
    # 这里的to只是让生成tensor的dytpe和device与boxes保持一致
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    # boxes加上对应层的偏移量后, 保证不同类别/层之间boxes不会有重合的现象
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


def remove_small_boxes(boxes: Tensor, min_size: float) -> Tensor:
    """
    移除宽/高小于min_size的boxes, 返回保留下来的boxes的索引

    参数:
    boxes: (N, 4)
    min_size: float

    返回:
    keep: (K, ), 返回的是保留的K个boxes的索引
    """
    # ws的形状: (N, )
    # hs的形状: (N, )
    ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    # keep的形状: (N, )
    # 满足条件(ws >= min_size) & (hs >= min_size)的元素为True, 否则为False
    keep = torch.logical_and(torch.ge(ws, min_size), torch.ge(hs, min_size))
    # keep的形状: (K, ) 对应的是为True的位子的索引
    keep = torch.where(keep)[0]
    return keep


def clip_boxes_to_image(boxes: Tensor, size: Tuple[int, int]) -> Tensor:
    """
    裁剪boxes, 将越界的坐标调整到图片边界(size)上

    参数:
    boxes: (N, 4)
    size: (H, W) padding前的图像尺寸

    返回:
    clipped_boxes: (N, 4)
    """
    dim = boxes.dim()
    # boxes_x的形状: (N, 2)
    boxes_x = boxes[..., 0::2]  # x1, x2
    # boxes_y的形状: (N, 2)
    boxes_y = boxes[..., 1::2]  # y1, y2
    height, width = size

    boxes_x = boxes_x.clamp(min=0, max=width)  # 限制x坐标范围在[0,width]之间
    boxes_y = boxes_y.clamp(min=0, max=height)  # 限制y坐标范围在[0,height]之间

    # clipped_boxes的形状: (N, 2, 2)
    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    # clipped_boxes的形状: (N, 4)
    return clipped_boxes.reshape(boxes.shape)


def box_area(boxes: Tensor) -> Tensor:
    """
    计算boxes的面积
    
    参数:
    boxes: (N, 4)

    返回:
    area: (N, )
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    IOU (Jaccard index) of boxes
    
    参数:
    boxes1: (N, 4)
    boxes2: (M, 4)

    返回:
    iou: (N, M)
    """
    # area1的形状: (N, )
    # area2的形状: (M, )
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # lt: (N, 1, 2), (M, 2) -> (N, M, 2)
    # rb: (N, 1, 2), (M, 2) -> (N, M, 2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom

    # wh: (N, M, 2)
    wh = (rb - lt).clamp(min=0)
    # inter: (N, M)
    inter = wh[:, :, 0] * wh[:, :, 1]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou