from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from det_utils import BoxCoder, Matcher, BalancedPositiveNegativeSampler, smooth_l1_loss
from box import box_iou, clip_boxes_to_image, remove_small_boxes, batched_nms


def fastrcnn_loss(class_logits: Tensor, box_regression: Tensor,
                  labels: List[Tensor],
                  regression_targets: List[Tensor]) -> Tuple[Tensor, Tensor]:
    """
    计算分类 & Offset的损失

    参数:
    class_logits: (M1+M2+..., num_classes)     - 预测类别logits
    box_regression: (M1+M2+..., num_classes*4) - 预测边目标界框offset
    labels: List[Tensor], Tensor的形状: (M, )   - 类别标签
    regression_targets: regression_targets: List[Tensor], Tensor的形状: (M, 4) - offsets标签

    返回: classification_loss, box_loss
    classification_loss: Tensor 标量
    box_loss: Tensor 标量
    """

    # labels的形状: (M1+M2+..., )
    labels = torch.cat(labels, dim=0)
    # regression_targets的形状: (M1+M2+..., 4)
    regression_targets = torch.cat(regression_targets, dim=0)

    # 计算类别损失信息
    classification_loss = F.cross_entropy(class_logits, labels)

    # sampled_pos_inds_subset的形状: (num_pos, ) - 正样本的位置的索引
    sampled_pos_inds_subset = torch.where(torch.gt(labels, 0))[0]

    # labels_pos的形状: (num_pos, ) - 正样本的类别
    labels_pos = labels[sampled_pos_inds_subset]

    N, _ = class_logits.shape  # N = M1+M2+...
    # box_regression的形状: (N, num_classes, 4)
    box_regression = box_regression.reshape(N, -1, 4)

    # 计算边界框损失信息
    box_loss = smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],  # (num_pos, 4)
        regression_targets[sampled_pos_inds_subset],  # (num_pos, 4)
        beta=1 / 9,
        size_average=False,
    ) / labels.numel()

    return classification_loss, box_loss


class RoIHeads(torch.nn.Module):
    __annotations__ = {
        'box_coder': BoxCoder,
        'proposal_matcher': Matcher,
        'fg_bg_sampler': BalancedPositiveNegativeSampler,
    }

    def __init__(self, box_roi_pool: nn.Module, box_head: nn.Module,
                 box_predictor: nn.Module, fg_iou_thresh: float,
                 bg_iou_thresh: float, batch_size_per_image: int,
                 positive_fraction: float,
                 bbox_reg_weights: Tuple[float, float, float,
                                         float], score_thresh: float,
                 nms_thresh: float, detection_per_img: int) -> None:
        """
        参数:
        box_roi_pool: MultiScaleRoIAlign
        box_head: TwoMLPHead. RoI Pooling/Align层的输出送入TwoMLPHead, 
                  针对每个proposals输出representation_size维的特征
        box_predictor: FastRCNNPredictor, TwoMLPHead层的输出送入FastRCNNPredictor, 
                       预测: `类别`和`offsets`
        fg_iou_thresh: 超过这个值会被标记为正样本 [Matcher]
        bg_iou_thresh: 低于这个值会被标记为负样本 [Matcher]
        batch_size_per_image: 计算损失时每张图片采用的正负样本的总个数 [BalancedPositiveNegativeSampler]
        positive_fraction: 正样本占计算损失时所有样本的比例 [BalancedPositiveNegativeSampler]
        bbox_reg_weights: 权重 [BoxCoder]
        score_thresh: 预测的proposals的概率小于这个会被移除
        nms_thresh: IoU>nms_thresh的boxes会被删除
        detection_per_img: 每张图片最多返回多少个预测的proposals
        """
        super(RoIHeads, self).__init__()

        self.box_similarity = box_iou
        # 根据IoU Matrix来标记proposals/anchors & gt-boxes
        self.proposal_matcher = Matcher(fg_iou_thresh,
                                        bg_iou_thresh,
                                        allow_low_quality_matches=False)

        # 对正负样本进行采样, 按照一定比例返回正样本和负样本
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(
            batch_size_per_image,  # 计算损失时每张图片采用的正负样本的总个数
            positive_fraction)  # 正样本占计算损失时所有样本的比例

        # Box Encoder & Decoder
        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool  # Multi-scale RoIAlign pooling
        self.box_head = box_head  # TwoMLPHead
        self.box_predictor = box_predictor  # FastRCNNPredictor

        self.score_thresh = score_thresh  # 预测的proposals的概率小于这个会被移除
        self.nms_thresh = nms_thresh
        self.detection_per_img = detection_per_img

    def assign_targets_to_proposals(
            self, proposals: List[Tensor], gt_boxes: List[Tensor],
            gt_labels: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """
        根据IoU Matrix来标记proposals & gt-boxes/gt-labels, 并划分为: 正样本, 背景以及废弃的样本

        labels:
        <1> 正样本处标记为类别, 正常类别从1开始
        <2> 负样本处标记为0, 也就是背景类
        <3> 丢弃样本处标记为-1
        
        参数:
        proposals: List[Tensor]
          List的元素个数=batch_size
          Tensor的形状: (K+N, 4)
          K为保留下来的proposals个数, 每个图像的K是不同的
          N为每张图片的gt boxes数量, 每张图像的K是不同的
        gt_boxes: gt_boxes: List[Tensor]
          Tensor的形状: (N, 4), N为每张图片的gt boxes数量, 每张图像的K是不同的
        gt_labels: gt_labels: List[List[Tensor]], Tensor是标量 - 标签列表(和gt boxes对应)
                              也就是每个gt box对应的类别
        返回: matched_idxs, labels
        matched_idxs: List[Tensor], Tensor的形状: (K+N,) - 每个proposals匹配到的gt-boxes索引
                      注意: 对于负样本/丢弃样本, 我们设置的gt-box索引为0, 这两类可以通过labels来区分
        labels: List[Tensor], Tensor的形状: (K+N,)       - 每个proposals匹配到的类别
          正样本处标记为类别, 正常类别从1开始
          负样本处标记为0, 也就是背景类
          丢弃样本处标记为-1
        """
        matched_idxs = []
        labels = []
        # 遍历每张图
        # proposals_in_image的形状: (K+N, 4)
        # gt_boxes_in_image的形状: (N, 4)
        # gt_labels_in_image的形状: List[Tensor], Tensor是标量, 元素数量为N
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(
                proposals, gt_boxes, gt_labels):
            if gt_boxes_in_image.numel() == 0:
                # 该张图像中没有gt box框, 所有proposals标记为负样本(背景) label=0
                device = proposals_in_image.device
                # clamped_matched_idxs_in_image的形状: (K+N, )
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0], ),
                    dtype=torch.int64,
                    device=device)
                # labels_in_image的形状: (K+N, )
                labels_in_image = torch.zeros((proposals_in_image.shape[0], ),
                                              dtype=torch.int64,
                                              device=device)
            else:
                # 计算proposals与真实gt boxes的iou信息
                # set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                # match_quality_matrix的形状:
                # (N, K+N)
                match_quality_matrix = box_iou(gt_boxes_in_image,
                                               proposals_in_image)

                # 标记proposals & gt-boxes
                # matched_idxs_in_image: (K+N, ) - 每个proposal对应的gt-box索引
                # <1> 正样本, 索引为: [0, N-1]
                # <2> iou小于low_threshold的matches索引置为: -1
                # <3> iou在[low_threshold, high_threshold]之间的matches索引置为: -2
                matched_idxs_in_image = self.proposal_matcher(
                    match_quality_matrix)

                # 注意:
                # 这里使用clamp设置下限0是为了方便取每个proposals对应的gt labels信息
                # 负样本和舍弃的样本都是负值, 为了防止越界直接置为0. 后续我们会进一步更新
                # 负样本和舍弃的样本对应的gt labels
                # clamped_matched_idxs_in_image的形状: (K+N, )
                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(
                    min=0)

                # labels_in_image记录所有proposals匹配后的类别标签
                # <1> 正样本处标记为类别, 正常类别从1开始
                # <2> 负样本处标记为0, 也就是背景类
                # <3> 丢弃样本处标记为-1
                #
                # 也就是每个proposals对应哪个类别: 0为背景; 正常类别从1开始
                # labels_in_image的形状: (K+N, )
                labels_in_image = gt_labels_in_image[
                    clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # 负样本(背景类)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD  # -1
                labels_in_image[bg_inds] = 0

                # 丢弃的样本
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS  # -2
                labels_in_image[ignore_inds] = -1

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels: List[Tensor]) -> List[Tensor]:
        """
        按给定`数量`和`比例`采样正负样本, 返回其索引
        比如: 一张图片采样256个样本, 其中正负样本的比例为1:3

        参数:
        labels: List[Tensor], Tensor的形状: (K+N,) - 每个proposals匹配到的类别
          正样本处标记为类别, 正常类别从1开始
          负样本处标记为0, 也就是背景类
          丢弃样本处标记为-1
        返回:
        sampled_inds: List[Tensor] - 采样出来的正负样本的索引
          Tensor的形状为: (M,) - M为正样本和负样本的数量和
        """
        # 对正负样本进行采样, 按照一定比例返回正样本和负样本
        # sampled_pos_inds: List[Tensor] - 正样本的mask
        #   Tensor的形状: (K+N, )
        #   正样本的位置会设置为1, 其它位置设置为0
        # sampled_neg_inds: List[Tensor] - 负样本的mask
        #   Tensor的形状: (K+N, )
        #   负样本的位置会设置为1, 其它位置设置为0
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        # 遍历每张图片的正/负样本mask
        # pos_inds_img的形状: (K+N, )
        # neg_inds_img的形状: (K+N, )
        for _, (pos_inds_img, neg_inds_img) in enumerate(
                zip(sampled_pos_inds, sampled_neg_inds)):
            # 记录所有采集样本索引(包括正样本和负样本)
            # img_sampled_inds的形状: (M, ) - M为正样本和负样本的数量和
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def add_gt_proposals(self, proposals: List[Tensor],
                         gt_boxes: List[Tensor]) -> List[Tensor]:
        """
        将每张图片的gt_boxes拼接到其proposals后面一起作为proposals

        参数:
        proposals: List[Tensor] - 最终保留下来的proposals
          List的元素个数=batch_size
          Tensor的形状: (K, 4), K为保留下来的proposals个数, 每个图像的K是不同的
        gt_boxes: gt_boxes: List[Tensor]
          Tensor的形状: (N, 4), N为每张图片的gt boxes数量, 每张图像的K是不同的

        返回:
        proposals: List[Tensor]
          List的元素个数=batch_size
          Tensor的形状: (K+N, 4)
          K为保留下来的proposals个数, 每个图像的K是不同的
          N为每张图片的gt boxes数量, 每张图像的K是不同的
        """
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]
        return proposals

    def check_targets(self, targets: List[Dict[str, Tensor]]) -> None:
        """
        参数:
        targets: list of dict, 每个dict包含如下k/v对:
          boxes    - list of [xmin, ymin, xmax, ymax]
          labels   - 标签列表
          image_id - 图片索引
          area     - 边界框面积
          iscrowd  - 表示目标检测的难易程度: 0表示容易检测; 1表示比较难检测
        """
        assert targets is not None
        assert all(["boxes" in t for t in targets])
        assert all(["labels" in t for t in targets])

    def select_training_samples(
        self, proposals: List[Tensor], targets: List[Dict[str, Tensor]]
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """
        从proposals中选择训练样本

        步骤:
        1. 将每张图片的gt_boxes拼接到其proposals后面一起作为proposals
        2. 根据IoU Matrix来标记proposals & gt-boxes/gt-labels, 并划分为:
           a. 正样本处标记为类别, 正常类别从1开始
           b. 负样本处标记为0, 也就是背景类
           c. 丢弃样本处标记为-1
           这一步结束后, 每一个proposals都会对应一个类别标签
        3. 按给定`数量`和`比例`采样正负样本, 返回其索引
           比如: 一张图片采样256个样本, 其中正负样本的比例为1:3
        4. 根据第3步给出的索引, 找出其对应的proposals, labels-[标签], gt boxes
        5. 根据proposals和其对应的gt boxes计算其对应的offsets-[标签]
        6. 返回最终的训练样本: proposals, labels-[标签], offsets-[标签]

        参数:
        proposals: List[Tensor] - 最终保留下来的proposals
          List的元素个数=batch_size
          Tensor的形状: (K, 4), K为保留下来的proposals个数, 每个图像的K是不同的
        targets: list of dict, 每个dict包含如下k/v对:
          boxes    - list of [xmin, ymin, xmax, ymax]
          labels   - 标签列表
          image_id - 图片索引
          area     - 边界框面积
          iscrowd  - 表示目标检测的难易程度: 0表示容易检测; 1表示比较难检测

        返回:  proposals, labels, regression_targets
        proposals: List[Tensor], Tensor的形状: (M, 4)          - 选出来的proposals
        labels: List[Tensor], Tensor的形状: (M, )              - 类别标签
        regression_targets: List[Tensor], Tensor的形状: (M, 4) - offsets标签
        """

        self.check_targets(targets)
        assert targets is not None
        dtype = proposals[0].dtype
        device = proposals[0].device

        # 获取标注好的boxes以及labels信息
        # gt_boxes: gt_boxes: List[Tensor]
        #  Tensor的形状: (N, 4), N为每张图片的gt boxes数量, 每张图像的N是不同的
        # gt_labels: List[List[Tensor]], Tensor是标量
        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # 将每张图片的gt_boxes拼接到其proposals后面一起作为proposals
        # proposals: List[Tensor]
        #   List的元素个数=batch_size
        #   Tensor的形状: (K+N, 4)
        #   K为保留下来的proposals个数, 每个图像的K是不同的
        #   N为每张图片的gt boxes数量, 每张图像的N是不同的
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # 根据IoU Matrix来标记proposals & gt-boxes/gt-labels
        # 并划分为: 正样本, 背景以及废弃的样本
        #
        # matched_idxs: List[Tensor], Tensor的形状: (K+N,) - 每个proposals匹配到的gt-boxes索引
        #               注意: 对于负样本/丢弃样本, 我们设置的gt-box索引为0, 这两类可以通过labels来区分
        # labels: List[Tensor], Tensor的形状: (K+N,)       - 每个proposals匹配到的类别
        #   正样本处标记为类别, 正常类别从1开始
        #   负样本处标记为0, 也就是背景类
        #   丢弃样本处标记为-1
        matched_idxs, labels = self.assign_targets_to_proposals(
            proposals, gt_boxes, gt_labels)
        # 按给定`数量`和`比例`采样正负样本, 返回其索引
        # 比如: 一张图片采样256个样本, 其中正负样本的比例为1:3
        # sampled_inds: List[Tensor] - 采样出来的正负样本的索引
        #  Tensor的形状为: (M,) - M为正样本和负样本的数量和
        sampled_inds = self.subsample(labels)

        matched_gt_boxes = []
        # num_images = batch_size
        num_images = len(proposals)

        # 遍历每张图像
        for img_id in range(num_images):
            # 获取图像的正负样本索引
            # img_sampled_inds的形状: (M, )
            img_sampled_inds = sampled_inds[img_id]
            # 获取正负样本的proposals信息
            # proposals[img_id]的形状: (K+N, 4) -> (M, 4)
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            # 获取正负样本的真实类别信息
            # labels[img_id]的形状: (K+N, ) -> (M, )
            labels[img_id] = labels[img_id][img_sampled_inds]
            # 获取对应正负样本的gt box索引信息
            # matched_idxs[img_id]的形状: (K+N, ) -> (M, )
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            # gt_boxes_in_image的形状: (N, 4)
            # N为每张图片的gt boxes数量, 每张图像的N是不同的
            gt_boxes_in_image = gt_boxes[img_id]

            # 对于没有gt box的图片, 构造一个索引为0的gt box
            # 因为负样本/丢弃样本, 默认关联到索引为0的gt-box
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4),
                                                dtype=dtype,
                                                device=device)
            # 获取对应正负样本的gt box信息
            # gt_boxes_in_image的形状: (M, 4)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        # matched_gt_boxes: List[Tensor], Tensor的形状: (M, 4)
        # proposals: List[Tensor], Tensor的形状: (M, 4)
        # regression_targets: List[Tensor], Tensor的形状: (M, 4)
        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)

        # proposals: List[Tensor], Tensor的形状: (M, 4)
        # labels: List[Tensor], Tensor的形状: (M, )
        # regression_targets: List[Tensor], Tensor的形状: (M, 4)
        return proposals, labels, regression_targets

    def postprocess_detections(
        self, class_logits: Tensor, box_regression: Tensor,
        proposals: List[Tensor], image_shapes: List[Tuple[int, int]]
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """
        
        步骤:
        1. 通过一个batch图片的offset和anchors/proposals的坐标, 计算其对应的boxes的坐标
           注意: 因为不同的类别有不同的offset, 所以不同的类别会有不同的boxes
        2. 对预测类别结果进行softmax处理
        3. 裁剪boxes, 将越界的坐标调整到图片边界(size)上
        4. 移除索引为0的预测信息(0代表背景)
        5. 移除小概率proposals(概率小于score_thresh)
        6. 移除宽/高小于min_size的proposals
        7. 针对不同类别, 单独进行NMS操作
        8. 根据scores排序返回前detection_per_img个目标
        9. 返回最终留下来的proposals及其对应的分数(概率值)

        参数:
        class_logits: (M1+M2+..., num_classes)                   - 网络预测类别概率
        box_regression: (M1+M2+..., num_classes*4)               - 网络预测的边界框offset
        proposals: List[Tensor], Tensor的形状: (M, 4)  - 选出来的proposals
        image_shapes: list of (H, W) padding前的图像尺寸, 一共batch_size个元素

        返回: all_boxes, all_scores, all_labels
        all_boxes: List[Tensor], Tensor的形状: (K, 4) - 最终保留下来的boxes, 按照分数从高到低降序排列
        all_scores: List[Tensor], Tensor的形状: (K,)  - 预测的分数
        all_labels: List[Tensor], Tensor的形状: (K,)  - 预测的类别标签
        """
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        # 获取每张图像的预测bbox数量
        boxes_per_image = [
            boxes_in_image.shape[0] for boxes_in_image in proposals
        ]
        # 通过一个batch图片的offset和anchors/proposals的坐标, 计算其对应的boxes的坐标
        # 注意: 因为不同的类别有不同的offset, 所以不同的类别会有不同的boxes
        # pred_boxes的的形状: (M1+M2+..., num_classes, 4)
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        # 对预测类别结果进行softmax处理
        # pred_scores的形状: (M1+M2+..., num_classes)
        pred_scores = F.softmax(class_logits, -1)

        # 根据每张图像的预测bbox数量分割结果
        # pred_boxes_list: List[Tensor], Tensor的形状: (M, num_classes, 4)
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        # pred_scores_list: List[Tensor], Tensor的形状: (M, num_classes)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        # 遍历每张图像预测信息
        # boxes的形状: (M, num_classes, 4)
        # scores的形状: (M, num_classes)
        # image_shape的形状: (H, W)
        for boxes, scores, image_shape in zip(pred_boxes_list,
                                              pred_scores_list, image_shapes):
            # 裁剪boxes, 将越界的坐标调整到图片边界(size)上
            # boxes的形状: (M, num_classes, 4)
            boxes = clip_boxes_to_image(boxes, image_shape)

            # labels的形状: (num_classes, )
            labels = torch.arange(num_classes, device=device)
            # labels的形状: (M, num_classes)
            labels = labels.view(1, -1).expand_as(scores)

            # 移除索引为0的预测信息(0代表背景)
            # boxes的形状:  (M, num_classes-1, 4)
            # scores的形状: (M, num_classes-1)
            # labels的形状: (M, num_classes-1)
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # boxes的形状:  (M*(num_classes-1), 4)
            # scores的形状: (M*(num_classes-1),)
            # labels的形状: (M*(num_classes-1),)
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # 移除小概率proposals(概率小于score_thresh)
            inds = torch.where(torch.gt(scores, self.score_thresh))[0]
            # boxes的形状:  (K, 4)
            # scores的形状: (K,)
            # labels的形状: (K,)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # 移除宽/高小于min_size的proposals
            keep = remove_small_boxes(boxes, min_size=1.)
            # boxes的形状:  (K, 4)
            # scores的形状: (K,)
            # labels的形状: (K,)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # 针对不同类别, 单独进行NMS操作
            # keep: (K,) 返回的是保留的boxes的索引, 按照分数从高到低降序排列
            keep = batched_nms(boxes, scores, labels, self.nms_thresh)

            # 根据scores排序返回前detection_per_img个目标
            keep = keep[:self.detection_per_img]
            # boxes的形状:  (K, 4)
            # scores的形状: (K,)
            # labels的形状: (K,)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def forward(
        self,
        features: Dict[str, Tensor],
        proposals: List[Tensor],
        image_shapes: List[Tuple[int, int]],
        targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]:
        """
        要注意预测模式和训练模式返回的结果是不同的
        
        参数:
        features: Dict[str, Tensor], e.g. {'0', Tensor0, '1', Tensor1, ...}
          经过backbone输出的feature map, 我们可以有多层features map特征, 有的backbone输出一层feature map,
          有的backbone输出多层feature map
          每个Tensor表示一层feature map, Tensor的形状(batch_size, 1280, h, w) 其中h/w是经过backbone处理
          之后的特征图的高和宽
        proposals: List[Tensor] - 最终保留下来的proposals
          List的元素个数=batch_size
          Tensor的形状: (K, 4), K为保留下来的proposals个数, 每个图像的K是不同的
        image_shapes: list of (H, W) padding前的图像尺寸, 一共batch_size个元素
        targets: list of dict, 每个dict包含如下k/v对:
          boxes    - list of [xmin, ymin, xmax, ymax]
          labels   - 标签列表
          image_id - 图片索引
          area     - 边界框面积
          iscrowd  - 表示目标检测的难易程度: 0表示容易检测; 1表示比较难检测
        
        返回: result: List[Dict[str, Tensor]], losses: Dict[str, Tensor]]
        result: - 测试模式下有数据
        List的元素个数是batch_size, dict的格式:
        {
            "boxes": Tensor的形状 (K, 4) - 最终保留下来的boxes, 按照分数从高到低降序排列
            "labels": Tensor的形状 (K,)  - 预测的分数
            "scores": Tensor的形状: (K,)  - 预测的类别标签
        }
        losses: - 训练模式下有数据
        {
            "loss_classifier": loss_classifier,
             "loss_box_reg": loss_box_reg
        }
        """
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t[
                    "boxes"].dtype in floating_point_types, "target boxes must of float type"
                assert t[
                    "labels"].dtype == torch.int64, "target labels must of int64 type"

        if self.training:
            # proposals: List[Tensor], Tensor的形状: (M, 4)          - 选出来的proposals
            # labels: List[Tensor], Tensor的形状: (M, )              - 类别标签
            # regression_targets: List[Tensor], Tensor的形状: (M, 4) - offsets标签
            proposals, labels, regression_targets = self.select_training_samples(
                proposals, targets)
        else:
            labels = None
            regression_targets = None

        # 将采集样本通过Multi-scale RoIAlign pooling层
        # box_features的形状: (M1+M2+..., 1280, 7, 7) - 假设RoI Align的输出为(7, 7)
        box_features = self.box_roi_pool(features, proposals, image_shapes)

        # RoI Pooling/Align层的输出送入TwoMLPHead, 针对每个proposals输出representation_size维的特征
        # box_features的形状: (M1+M2+..., representation_size)
        box_features = self.box_head(box_features)

        # TwoMLPHead层的输出送入FastRCNNPredictor, 预测: `类别`和`offsets`
        # class_logits: (M1+M2+..., num_classes)
        # box_regression: (M1+M2+..., num_classes*4)
        class_logits, box_regression = self.box_predictor(box_features)

        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            # 计算分类 & Offset的损失
            # classification_loss: Tensor 标量
            # box_loss: Tensor 标量
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        else:
            # boxes: List[Tensor], Tensor的形状: (K, 4) - 最终保留下来的boxes, 按照分数从高到低降序排列
            # scores: List[Tensor], Tensor的形状: (K,)  - 预测的分数
            # labels: List[Tensor], Tensor的形状: (K,)  - 预测的类别标签
            boxes, scores, labels = self.postprocess_detections(
                class_logits, box_regression, proposals, image_shapes)
            # num_images = batch_size
            num_images = len(boxes)
            for i in range(num_images):
                result.append({
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                })

        return result, losses