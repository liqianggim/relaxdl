from collections import OrderedDict
from typing import Tuple, List, Dict, Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign

from roi_head import RoIHeads
from transform import GeneralizedRCNNTransform
from rpn import AnchorsGenerator, RPNHead, RegionProposalNetwork
from mobilenet_v2 import MobileNetV2


class FasterRCNNBase(nn.Module):
    def __init__(self, backbone: MobileNetV2, rpn: RegionProposalNetwork,
                 roi_heads: RoIHeads,
                 transform: GeneralizedRCNNTransform) -> None:
        """
        参数: 
        backbone: 默认为: MobileNetV2. 用来抽取特征. 需要有out_channels这个属性, 表示每一层
                  feature map的输出通道数(如果有多层feature map, 每一层的输出通道都需要是out_channels). 
                  backbone需要返回Tensor或者是OrderedDict[Tensor]
        rpn: Region Proposal Network
        roi_heads: RoI Heads
        transform: 将load_pascal_voc(batch_size)获得的一个batch的images和target做Transform, 
                   处理之后的结果可以送入Faster RCNN网络
        """
        super(FasterRCNNBase, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False

    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        """        
        参数:
        images: List[Tensor], 每个Tensor为一张图片, 形状为: (C,H,W)
        targets: List[Dict[str, Tensor]]
          每个dict包含如下k/v对:
            boxes    - list of [xmin, ymin, xmax, ymax]
            labels   - 标签列表
            image_id - 图片索引
            area     - 边界框面积
            iscrowd  - 表示目标检测的难易程度: 0表示容易检测; 1表示比较难检测

        返回: losses:
        losses: Dict[str, Tensor]
        { 
             "loss_objectness": RPN分类损失
             "loss_rpn_box_reg": RPN回归损失
             "loss_classifier": R-CNN分类损失
             "loss_box_reg": R-CNN回归损失
        }
        detections: List[Dict[str, Tensor]] - 测试模式下有数据
        List的元素个数是batch_size, dict的格式:
        {
             "boxes": Tensor的形状 (K, 4) - 最终保留下来的boxes, 按照分数从高到低降序排列
             "labels": Tensor的形状 (K,)  - 预测的分数
             "scores": Tensor的形状: (K,)  - 预测的类别标签
        }
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        # List[(h, w)] - 保存的是每张图片的原始尺寸
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        # images:
        #   ImageList: 包含下面两部分信息
        #   a. ImageList.tensors: padding后的图像数据, 图像具有相同的尺寸: (batch_size, C, H_new, W_new)
        #   b. ImageList.image_size: list of (H, W) padding前的图像尺寸, 一共batch_size个元素
        # targets: list of dict, 每个dict包含如下k/v对:
        #     boxes    - list of [xmin, ymin, xmax, ymax]
        #     labels   - 标签列表
        #     image_id - 图片索引
        #     area     - 边界框面积
        #     iscrowd  - 表示目标检测的难易程度: 0表示容易检测; 1表示比较难检测
        images, targets = self.transform(images, targets)  # 对图像进行预处理

        # 我们使用MobileNet V2作为backbone, 输出只有一层feature map
        # features的形状: (batch_size, 1280, h, w) - 其中h/w是feature map的尺寸
        features = self.backbone(images.tensors)

        # features: Dict[str, Tensor], e.g. {'0', Tensor0, '1', Tensor1, ...}
        #   经过backbone输出的feature map, 我们可以有多层features map特征, 有的backbone输出一层feature map,
        #   有的backbone输出多层feature map
        #   每个Tensor表示一层feature map, Tensor的形状(batch_size, 1280, h, w) 其中h/w是经过backbone处理
        #   之后的特征图的高和宽
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        # proposals: List[Tensor] - 最终保留下来的proposals
        #   List的元素个数=batch_size
        #   Tensor的形状: (K, 4), K为保留下来的proposals个数, 每个图像的K是不同的
        # proposal_losses: Dict[str, Tensor], 训练模式有losses, 测试模式没有losses
        #   {
        #     "loss_objectness": loss_objectness,
        #     "loss_rpn_box_reg": loss_rpn_box_reg
        #   }
        proposals, proposal_losses = self.rpn(images, features, targets)

        # detections:List[Dict[str, Tensor]] - 测试模式下有数据
        # List的元素个数是batch_size, dict的格式:
        # {
        #     "boxes": Tensor的形状 (K, 4) - 最终保留下来的boxes, 按照分数从高到低降序排列
        #     "labels": Tensor的形状 (K,)  - 预测的分数
        #     "scores": Tensor的形状: (K,)  - 预测的类别标签
        # }
        # detector_losses:Dict[str, Tensor]] - 训练模式下有数据
        # {
        #     "loss_classifier": loss_classifier,
        #      "loss_box_reg": loss_box_reg
        # }
        detections, detector_losses = self.roi_heads(features, proposals,
                                                     images.image_sizes,
                                                     targets)

        # 对网络的预测结果进行后处理(将bboxes还原到原图像尺度上)
        # detections:List[Dict[str, Tensor]] - 测试模式下有数据
        # List的元素个数是batch_size, dict的格式:
        # {
        #     "boxes": Tensor的形状 (K, 4) - 最终保留下来的boxes, 按照分数从高到低降序排列
        #     "labels": Tensor的形状 (K,)  - 预测的分数
        #     "scores": Tensor的形状: (K,)  - 预测的类别标签
        # }
        detections = self.transform.postprocess(detections, images.image_sizes,
                                                original_image_sizes)

        losses = {}
        # {
        #     "loss_objectness": 属于proposal_losses
        #     "loss_rpn_box_reg": 属于proposal_losses
        #     "loss_classifier": 属于detector_losses
        #     "loss_box_reg": 属于detector_losses
        # }
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return losses, detections


class TwoMLPHead(nn.Module):
    """
    RoI Pooling/Align层的输出送入TwoMLPHead, 针对每个proposals输出representation_size维的特征
    """
    def __init__(self, in_channels: int, representation_size: int) -> None:
        """
        参数: 
        in_channels: 输入的尺寸
        representation_size: 输出的尺寸
        """
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        参数:
        x: (N, C, H, W) | (N, C*H*W)

        返回:
        outputs: (N, representation_size)
        """
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class FastRCNNPredictor(nn.Module):
    """
    TwoMLPHead层的输出送入FastRCNNPredictor, 预测: `类别`和`offsets`
    """
    def __init__(self, in_channels: int, num_classes: int) -> None:
        """
        参数: 
        in_channels: 输入的尺寸
        num_classes: 类别的数量(每个类别都会预测offset)
        """
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        参数:
        x: (N, C, 1, 1) | (N, C)

        返回: scores, bbox_deltas
        scores: (N, num_classes)
        bbox_deltas: (N, num_classes*4)
        """
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)  # 类别
        bbox_deltas = self.bbox_pred(x)  # offsets

        return scores, bbox_deltas


class FasterRCNN(FasterRCNNBase):
    """
    实现Faster R-CNN

    模型的输入是List[Tensor], 每个Tensor的形状是[C,H,W], 不同的images的尺寸可以是不同的. 模块在训练模式和测试模式
    下的行为是不同的.

    训练模式:
    模型需要输入Tensors, 同时需要输入targets, 每个图片一个dict, 需要包含:
        - boxes(FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, 
          with values between 0 and H and 0 and W
        - labels(Int64Tensor[N]): the class label for each ground-truth box
    模型在训练模式下会返回Dict[str, Tensor]格式的Loss, 包含了RPN和R-CNN的loss
        - loss_objectness: RPN分类损失
        - loss_rpn_box_reg: RPN回归损失
        - loss_classifier: R-CNN分类损失
        - loss_box_reg: R-CNN回归损失

    测试模式:
    模型只需要输入Tensors, 不需要输入targets, 返回预测的结果detections, 每个图片对应一个dict, 格式如下:
        - boxes(FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, 
          with values between0 and H and 0 and W
        - labels(Int64Tensor[N]): the predicted labels for each image
        - scores(Tensor[N]): the scores or each prediction
    """
    def __init__(
            self,
            backbone: MobileNetV2,
            num_classes: int = None,
            min_size: int = 800,
            max_size: int = 1333,
            image_mean: Tuple[float, float, float] = None,
            image_std: Tuple[float, float, float] = None,
            rpn_anchor_generator: AnchorsGenerator = None,
            rpn_head: RPNHead = None,
            rpn_pre_nms_top_n_train: int = 2000,
            rpn_pre_nms_top_n_test: int = 1000,
            rpn_post_nms_top_n_train: int = 2000,
            rpn_post_nms_top_n_test: int = 1000,
            rpn_nms_thresh: float = 0.7,
            rpn_fg_iou_thresh: float = 0.7,
            rpn_bg_iou_thresh: float = 0.3,
            rpn_batch_size_per_image: int = 256,
            rpn_positive_fraction: float = 0.5,
            rpn_score_thresh: float = 0.0,
            box_roi_pool: MultiScaleRoIAlign = None,
            box_head: TwoMLPHead = None,
            box_predictor: FastRCNNPredictor = None,
            box_score_thresh=0.05,
            box_nms_thresh=0.5,
            box_detections_per_img=100,
            box_fg_iou_thresh=0.5,
            box_bg_iou_thresh=0.5,  # fast rcnn计算误差时，采集正负样本设置的阈值
            box_batch_size_per_image=512,
            box_positive_fraction=0.25,  # fast rcnn计算误差时采样的样本数，以及正样本占所有样本的比例
            bbox_reg_weights=None):
        """
        参数:
        backbone: 默认为: MobileNetV2. 用来抽取特征. 需要有out_channels这个属性, 表示每一层
                  feature map的输出通道数(如果有多层feature map, 每一层的输出通道都需要是out_channels). 
                  backbone需要返回Tensor或者是OrderedDict[Tensor]
        num_classes: 类别. 如果设置了box_predictor参数, 则类别为None

        # transform parameter
        min_size: 图像在送入backbone之间会被处理成的最小边长
        max_size: 图像在送入backbone之间会被处理成的最大边长
        image_mean: 图像在标准化处理中的均值 e.g. [0.485, 0.456, 0.406]
        image_std: 图像在标准化处理中的方差 e.g. [0.229, 0.224, 0.225]

        # RPN parameters
        rpn_anchor_generator: 将`一个批量的图片imageList`以及`其backbone输出的feature_maps`作为输入, 
                              输出`每张图片对应的anchors(是由feature map上的anchors映射过去的)`
        rpn_head: 将backbone输出的feature map送入RPNHead, RPNHead会输出feature map上每个像素对应的
                  anchors的`分类`和`offset结果`
        rpn_pre_nms_top_n_train: 在NMS处理前保留的proposal数(training模式)
        rpn_pre_nms_top_n_test: 在NMS处理前保留的proposal数(testing模式)
        rpn_post_nms_top_n_train: 在NMS处理后保留的proposal数(training模式)
        rpn_post_nms_top_n_test: 在NMS处理后保留的proposal数(testing模式)
        rpn_nms_thresh: 进行NMS处理时使用的IoU阈值, 大于这个阈值的会被删除
        rpn_fg_iou_thresh: 如果生成的anchors和ground truth的IoU大于这个值, 则标记为正样本. e.g. 0.7
        rpn_bg_iou_thresh: 如果生成的anchors和ground truth的IoU小于这个值, 则标记为负样本, e.g. 0.3
        rpn_batch_size_per_image: 计算损失时每张图片采用的正负样本的总个数, e.g. 256
        rpn_positive_fraction: 正样本占计算损失时所有样本的比例, e.g. 0.5
        rpn_score_thresh: 预测的proposals的概率小于这个会被移除

        # Module parameters
        box_roi_pool: MultiScaleRoIAlign
        box_head: TwoMLPHead. RoI Pooling/Align层的输出送入TwoMLPHead, 
                  针对每个proposals输出representation_size维的特征
        box_predictor: FastRCNNPredictor, TwoMLPHead层的输出送入FastRCNNPredictor, 
                       预测: `类别`和`offsets`

        # RoIHeads parameters
        box_score_thresh: RoIHeads - 预测的proposals的概率小于这个会被移除
        box_nms_thresh: RoIHeads - IoU>nms_thresh的boxes会被删除
        box_detections_per_img:  RoIHeads - 每张图片最多返回多少个预测的proposals
        box_fg_iou_thresh:  RoIHeads - 超过这个值会被标记为正样本 [Matcher]
        box_bg_iou_thresh:  RoIHeads -   低于这个值会被标记为负样本 [Matcher]
        box_batch_size_per_image:  RoIHeads - 计算损失时每张图片采用的正负样本的总个数 [BalancedPositiveNegativeSampler]
        box_positive_fraction: RoIHeads - 正样本占计算损失时所有样本的比例 [BalancedPositiveNegativeSampler]
        bbox_reg_weights: RoIHeads - 权重 [BoxCoder]
        """
        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels"
                "specifying the number of output channels  (assumed to be the"
                "same for all the levels")

        assert isinstance(rpn_anchor_generator, (AnchorsGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError(
                    "num_classes should be None when box_predictor "
                    "is specified")
        else:
            if box_predictor is None:
                raise ValueError(
                    "num_classes should not be None when box_predictor "
                    "is not specified")

        # 预测特征层的channels
        out_channels = backbone.out_channels

        # 若anchor生成器为空，则自动生成针对resnet50_fpn的anchor生成器
        if rpn_anchor_generator is None:
            anchor_sizes = ((32, ), (64, ), (128, ), (256, ), (512, ))
            aspect_ratios = ((0.5, 1.0, 2.0), ) * len(anchor_sizes)
            rpn_anchor_generator = AnchorsGenerator(anchor_sizes,
                                                    aspect_ratios)

        # 生成RPN通过滑动窗口预测网络部分
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels,
                rpn_anchor_generator.num_anchors_per_location()[0])

        # 默认rpn_pre_nms_top_n_train = 2000, rpn_pre_nms_top_n_test = 1000,
        # 默认rpn_post_nms_top_n_train = 2000, rpn_post_nms_top_n_test = 1000,
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train,
                                 testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train,
                                  testing=rpn_post_nms_top_n_test)

        # 定义整个RPN框架
        rpn = RegionProposalNetwork(rpn_anchor_generator,
                                    rpn_head,
                                    rpn_fg_iou_thresh,
                                    rpn_bg_iou_thresh,
                                    rpn_batch_size_per_image,
                                    rpn_positive_fraction,
                                    rpn_pre_nms_top_n,
                                    rpn_post_nms_top_n,
                                    rpn_nms_thresh,
                                    score_thresh=rpn_score_thresh)

        #  Multi-scale RoIAlign pooling
        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],  # 在哪些特征层进行roi pooling
                output_size=[7, 7],
                sampling_ratio=2)

        # fast RCNN中roi pooling后的展平处理两个全连接层部分
        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            # out_channels: 是RoI Pooling层输出的通道数, 在MobileNet V2中是1280
            # resolution: 默认等于7, 也就是RoI Pooling的输出是: (7,7)
            # representation_size: 表示TwoMLPHead最终输出的维度
            box_head = TwoMLPHead(out_channels * resolution**2,
                                  representation_size)

        # 在box_head的输出上预测部分
        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(representation_size, num_classes)

        # 将roi pooling, box_head以及box_predictor结合在一起
        roi_heads = RoIHeads(
            # box
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,  # 0.5  0.5
            box_batch_size_per_image,
            box_positive_fraction,  # 512  0.25
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img)  # 0.05  0.5  100

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        # 对数据进行标准化，缩放，打包成batch等处理部分
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean,
                                             image_std)

        super(FasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)