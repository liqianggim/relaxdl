from typing import Tuple, List
import math
import torch
from torch import Tensor


class BalancedPositiveNegativeSampler(object):
    """
    对正负样本进行采样, 按照一定比例返回正样本和负样本
    """
    def __init__(self, batch_size_per_image: int,
                 positive_fraction: float) -> None:
        """
        参数:
        batch_size_per_image: 计算损失时每张图片采用的正负样本的总个数, e.g. 256
        positive_fraction: 正样本占计算损失时所有样本的比例, e.g. 0.5
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(
            self,
            matched_idxs: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """
        对正负样本进行采样, 按照一定比例返回正样本和负样本

        参数:
        matched_idxs: 真实的标签 List[Tensor] - 为每个anchor分配的标签
          Tensor形状为: (N, )
          a. 正样本>=1
          b. 负样本(背景)=0
          c. 废弃的样本=-1

        返回: pos_idx, neg_idx
        pos_idx: List[Tensor] - 正样本的mask
          Tensor的形状: (N, )
          正样本的位置会设置为1, 其它位置设置为0
        neg_idx: List[Tensor] - 负样本的mask
          Tensor的形状: (N, )
          负样本的位置会设置为1, 其它位置设置为0
        """
        pos_idx = []
        neg_idx = []
        # 遍历每张图像
        # matched_idxs_per_image的形状: (N, )
        for matched_idxs_per_image in matched_idxs:
            # positive - (num_positive, ) 正样本(>=1)的位置索引
            positive = torch.where(torch.ge(matched_idxs_per_image, 1))[0]
            # negative - (num_negative, ) 负样本(=0)的位置索引
            negative = torch.where(torch.eq(matched_idxs_per_image, 0))[0]

            # 计算正样本和负样本的数量
            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            num_neg = min(negative.numel(), num_neg)

            # 将正样本和负样本的索引打乱顺序
            # perm1 - (num_pos, )
            # perm2 - (num_neg, )
            perm1 = torch.randperm(positive.numel(),
                                   device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(),
                                   device=negative.device)[:num_neg]
            # pos_idx_per_image - (num_pos, ) 正样本的位置索引
            pos_idx_per_image = positive[perm1]
            # neg_idx_per_image - (num_neg, ) 负样本的位置索引
            neg_idx_per_image = negative[perm2]

            # pos_idx_per_image_mask的形状: (N, )
            # 正样本的位置会设置为1, 其它位置设置为0
            pos_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8)  # 注意: mask类型
            # neg_idx_per_image_mask的形状: (N, )
            # 负样本的位置会设置为1, 其它位置设置为0
            neg_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8)  # 注意: mask类型

            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx


@torch.jit._script_if_tracing
def encode_boxes(reference_boxes: Tensor, proposals: Tensor,
                 weights: Tensor) -> Tensor:
    """
    根据proposals及其对应的gt box计算offset

    参数:
    reference_boxes: 与proposals匹配的gt box
        (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 4)
    proposals: 每张图片对应的proposals
        (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 4)
    weights: Tuple[float, float, float, float] e.g.(1.0, 1.0, 1.0, 1.0)
    
    返回:
    targets: (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 4)
    """

    # perform some unpacking to make it JIT-fusion friendly
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]

    # proposals_x1的形状: (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 1)
    # proposals_y1的形状: (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 1)
    # proposals_x2的形状: (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 1)
    # proposals_y2的形状: (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 1)
    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)

    # reference_boxes_x1的形状: (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 1)
    # reference_boxes_y1的形状: (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 1)
    # reference_boxes_x2的形状: (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 1)
    # reference_boxes_y2的形状: (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 1)
    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

    # ex_widths的形状: (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 1)
    # ex_heights的形状: (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 1)
    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1
    # ex_ctr_x的形状: (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 1)
    # ex_ctr_y的形状: (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 1)
    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights

    # gt_widths的形状: (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 1)
    # gt_heights的形状: (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 1)
    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    # gt_ctr_x的形状: (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 1)
    # gt_ctr_x的形状: (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 1)
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

    # targets_dx的形状: (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 1)
    # targets_dy的形状: (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 1)
    # targets_dw的形状: (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 1)
    # targets_dh的形状: (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 1)
    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)

    # targets的形状: (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 4)
    targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh),
                        dim=1)
    return targets


class BoxCoder(object):
    """
    Box Encoder & Decoder
    1. encode - 根据proposals/anchors和其匹配的gt boxes计算offset [可以作为训练时的label使用]
    2. decode - 根据offset和其匹配的anchors的坐标, 计算其对应的boxes的坐标
    """
    def __init__(
        self,
        weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        bbox_xform_clip: float = math.log(1000. / 16)
    ) -> None:
        """
        参数:
        weights: 权重
        bbox_xform_clip: 限制max value, 防止exp()的时候数值太大
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes: List[Tensor],
               proposals: List[Tensor]) -> List[Tensor]:
        """
        根据proposals/anchors和其匹配的gt boxes计算offset [可以作为训练时的label使用]

        参数:
        reference_boxes: List[Tensor] - 与proposals匹配的gt box
          Tensor形状为: (N, 4)
        proposals: List[Tensor], 每张图片对应的proposals
          1. 第一层List是这个batch有多少张图片, 也就是batch_size
          2. Tensor的形状: (N, 4)
        
        返回:
        offsets: List[Tensor]
          Tensor的形状: (N, 4)
        """
        # boxes_per_image: List[int], 记录了每张图片有多少个proposals
        # 方便后续的split操作
        boxes_per_image = [len(b) for b in reference_boxes]
        # reference_boxes的形状: (batch_size*N, 4)
        reference_boxes = torch.cat(reference_boxes, dim=0)
        # proposals的形状: (batch_size*N, 4)
        proposals = torch.cat(proposals, dim=0)

        #  targets: (batch_size*N, 4)
        targets = self.encode_single(reference_boxes, proposals)

        # List[Tensor]
        # Tensor的形状: (N, 4)
        return list(targets.split(boxes_per_image, 0))

    def encode_single(self, reference_boxes, proposals):
        """
        根据proposals及其对应的gt box计算offset

        参数:
        reference_boxes: 与proposals匹配的gt box
            (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 4)
        proposals: 每张图片对应的proposals
            (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 4)
        weights: Tuple[float, float, float, float] e.g.(1.0, 1.0, 1.0, 1.0)
        
        返回:
        targets: (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 4)
        """
        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        # targets: (batch_size*(h1*w1*num_anchors1+h2*w2*num_anchors2+...), 4)
        targets = encode_boxes(reference_boxes, proposals, weights)

        return targets

    def decode(self, rel_codes: Tensor, boxes: List[Tensor]) -> Tensor:
        """    
        通过一个batch图片的offset和anchors/proposals的坐标, 计算其对应的boxes的坐标
        注意: 因为不同的类别有不同的offset, 所以不同的类别会有不同的boxes
    
        参数:
        rel_codes的形状: (M1+M2+..., num_classes*4) 预测的offset
        boxes: List[Tensor], 每张图片对应的anchors/proposals
          1. 第一层List是这个batch有多少张图片, 也就是batch_size
          2. Tensor的形状: (M, 4)
        
        返回:
        pred_boxes的形状: (M1+M2+..., num_classes, 4)
        """
        assert isinstance(boxes, (list, tuple))
        assert isinstance(rel_codes, torch.Tensor)
        # boxes_per_image: List[int]
        # 1. 数量是batch_size
        # 2. 里面的元素: M1, M2, ...
        boxes_per_image = [b.size(0) for b in boxes]
        # concat_boxes的形状: (M1+M2+..., 4)
        concat_boxes = torch.cat(boxes, dim=0)

        # 一个批量的box总数:
        # M1+M2+...
        box_sum = 0
        for val in boxes_per_image:
            box_sum += val

        # 将预测的bbox offset应用到对应anchors/proposals上得到预测bbox的坐标
        # rel_codes的形状: (M1+M2+..., num_classes*4) - 预测的offset
        # concat_boxes的形状: (M1+M2+..., 4) - 其对应的anchors
        # pred_boxes的形状: (M1+M2+..., num_classes*4)
        pred_boxes = self.decode_single(rel_codes, concat_boxes)

        # 防止pred_boxes为空时导致reshape报错
        if box_sum > 0:
            # pred_boxes的形状: (M1+M2+..., num_classes, 4)
            pred_boxes = pred_boxes.reshape(box_sum, -1, 4)

        return pred_boxes

    def decode_single(self, rel_codes, boxes):
        """
        将预测的offset应用到对应anchors上得到预测boxes的坐标

        参数:
        rel_codes的形状: (M1+M2+..., num_classes*4) - offset
        boxes的形状: (M1+M2+..., 4) - anchor/proposal

        返回:
        pred_boxes的形状: (M1+M2+..., num_classes*4)
        """
        boxes = boxes.to(rel_codes.dtype)

        # xmin, ymin, xmax, ymax
        # widths的形状:  (M1+M2+..., )
        # heights的形状: (M1+M2+..., )
        # ctr_x的形状: (M1+M2+..., )
        # ctr_y的形状: (M1+M2+..., )
        widths = boxes[:, 2] - boxes[:, 0]  # anchor/proposal宽度
        heights = boxes[:, 3] - boxes[:, 1]  # anchor/proposal高度
        ctr_x = boxes[:, 0] + 0.5 * widths  # anchor/proposal中心x坐标
        ctr_y = boxes[:, 1] + 0.5 * heights  # anchor/proposal中心y坐标

        wx, wy, ww, wh = self.weights
        # dx的形状: (M1+M2+..., num_classes)
        # dy的形状: (M1+M2+..., num_classes)
        # dw的形状: (M1+M2+..., num_classes)
        # dh的形状: (M1+M2+..., num_classes)
        dx = rel_codes[:, 0::4] / wx  # 预测anchors/proposals的中心坐标x回归参数
        dy = rel_codes[:, 1::4] / wy  # 预测anchors/proposals的中心坐标y回归参数
        dw = rel_codes[:, 2::4] / ww  # 预测anchors/proposals的宽度回归参数
        dh = rel_codes[:, 3::4] / wh  # 预测anchors/proposals的高度回归参数

        # limit max value, prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        # pred_ctr_x的形状: (M1+M2+..., num_classes)
        # pred_ctr_y的形状: (M1+M2+..., num_classes)
        # pred_w的形状: (M1+M2+..., num_classes)
        # pred_h的形状: (M1+M2+..., num_classes)
        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        # xmin的形状: (M1+M2+..., num_classes)
        pred_boxes1 = pred_ctr_x - torch.tensor(
            0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        # ymin的形状: (M1+M2+..., num_classes)
        pred_boxes2 = pred_ctr_y - torch.tensor(
            0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        # xmax的形状: (M1+M2+..., num_classes)
        pred_boxes3 = pred_ctr_x + torch.tensor(
            0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        # ymax的形状: (M1+M2+..., num_classes)
        pred_boxes4 = pred_ctr_y + torch.tensor(
            0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h

        # pred_boxes的形状:
        # (M1+M2+..., num_classes, 4) -> (M1+M2+..., num_classes*4)
        pred_boxes = torch.stack(
            (pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4),
            dim=2).flatten(1)
        return pred_boxes


class Matcher(object):
    """
    根据IoU Matrix来标记anchors & gt-boxes
    1. iou>high_threshold的标记为`正样本`, 索引为: [0, M-1]
    2. iou小于low_threshold的标记为`负样本`, 索引为: -1
    3. iou在[low_threshold, high_threshold]之间的样本`丢弃`, 索引置为: -2
    """
    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    __annotations__ = {
        'BELOW_LOW_THRESHOLD': int,
        'BETWEEN_THRESHOLDS': int,
    }

    def __init__(self,
                 high_threshold: float,
                 low_threshold: float,
                 allow_low_quality_matches: bool = False) -> None:
        """
        参数:
        high_threshold: 超过这个值会被标记为正样本
        low_threshold: 低于这个值会被标记为负样本
        allow_low_quality_matches: 如果为True, 对于每一个gt box, 我们至少会给他匹配上一个anchor
        """
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold  # 0.7
        self.low_threshold = low_threshold  # 0.3
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix: Tensor) -> Tensor:
        """
        根据IoU Matrix来标记anchors & gt-boxes
        
        1. iou>high_threshold的标记为`正样本`, 索引为: [0, M-1]
        2. iou小于low_threshold的标记为`负样本`, 索引为: -1
        3. iou在[low_threshold, high_threshold]之间的样本`丢弃`, 索引置为: -2

        参数:
        match_quality_matrix: (M, N) IOU Matrix
         M - gt boxes的数量; N - anchors的数量

        返回:
        matches: (N, ) - 每个anchor对应的gt-box索引
        """
        if match_quality_matrix.numel() == 0:
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    "No ground-truth boxes available for one of the images "
                    "during training")
            else:
                raise ValueError(
                    "No proposal boxes available for one of the images "
                    "during training")

        # 针对每个anchors, 找最大IoU的gt-box
        # matched_vals的形状: (N,) - 每个anchor对应的IoU最大的gt boxe的IoU
        # matches的形状: (N,)      - 每个anchor对应的IoU最大的gt boxe的索引
        matched_vals, matches = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        else:
            all_matches = None

        # 1. 计算iou小于low_threshold的索引
        # below_low_threshold的形状: (N, ), 里面的元素是True|False
        below_low_threshold = matched_vals < self.low_threshold
        # 2. 计算iou在low_threshold与high_threshold之间的索引值
        # between_thresholds的形状: (N, ), 里面的元素是True|False
        between_thresholds = (matched_vals >= self.low_threshold) & (
            matched_vals < self.high_threshold)

        # 更新matches中的数据:
        # 1. iou>high_threshold的标记为`正样本`, 索引为: [0, M-1]
        # 2. iou小于low_threshold的标记为`负样本`, 索引为: -1
        # 3. iou在[low_threshold, high_threshold]之间的样本`丢弃`, 索引置为: -2
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD  # -1
        matches[between_thresholds] = self.BETWEEN_THRESHOLDS  # -2

        if self.allow_low_quality_matches:
            assert all_matches is not None
            # 这个方法会更新matches(对于每一个gt box, 我们至少会给他匹配上一个anchor)
            self.set_low_quality_matches_(matches, all_matches,
                                          match_quality_matrix)

        return matches

    def set_low_quality_matches_(self, matches: Tensor, all_matches: Tensor,
                                 match_quality_matrix: Tensor) -> None:
        """
        这个方法会更新matches(对于每一个gt box, 我们至少会给他匹配上一个anchor)

        参数:
        matches: (N,)
        <1> iou>high_threshold的标记为`正样本`, 索引为: [0, M-1]
        <2> iou小于low_threshold的标记为`负样本`, 索引为: -1
        <3> iou在[low_threshold, high_threshold]之间的样本`丢弃`, 索引置为: -2
        all_matches: (N,) - 每个anchor对应的IoU最大的gt boxe的索引
        match_quality_matrix: (M, N)
         M - gt boxes的数量; N - anchors的数量
        
        返回:
        None
        """
        # 针对每个gt-box, 找出其IoU最大的anchors
        # highest_quality_foreach_gt的形状: (M, ) - 每个gt box对应的IoU最大的anchor的IoU
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)

        # 寻找每个gt boxe与其iou最大的anchor索引, 要注意: 一个gt匹配到的最大iou可能有多个anchor
        # gt_pred_pairs_of_highest_quality: Tuple[Tensor, Tensor]
        # 2个Tensor的形状都是: (K, ), 其中 K >= M
        # 第一个Tensor取值范围: [0, M-1]
        # 第二个Tensor取值范围: [0, N-1]
        gt_pred_pairs_of_highest_quality = torch.where(
            torch.eq(match_quality_matrix, highest_quality_foreach_gt[:,
                                                                      None]))

        # gt_pred_pairs_of_highest_quality[:, 0]代表是对应的gt index(不需要)
        # pre_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
        pre_inds_to_update = gt_pred_pairs_of_highest_quality[1]
        # 保留该anchor匹配gt最大iou的索引，即使iou低于设定的阈值
        matches[pre_inds_to_update] = all_matches[pre_inds_to_update]


def smooth_l1_loss(input: Tensor,
                   target: Tensor,
                   beta: float = 1. / 9,
                   size_average: bool = True) -> Tensor:
    """
    参数:
    input: (N, 4)  预测值
    target: (N, 4) 标签
    beta: float
    size_average: 是否返回均值

    返回:
    Tensor - 标量
    """
    n = torch.abs(input - target)
    cond = torch.lt(n, beta)
    loss = torch.where(cond, 0.5 * n**2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()
