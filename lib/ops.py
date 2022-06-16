import math
import torch
from torch import nn, jit, Tensor
from typing import List, Tuple
from torchvision.extension import _assert_has_ops


def compute_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    # 【Batch IOU】IOU计算的简单理解 https://zhuanlan.zhihu.com/p/424241927
    # https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]

    wh = (rb-lt).clamp(min=0)     # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    union = area1[:, None] + area2[None, :] - inter  # [N, M]
    iou = inter / union
    return iou


def nms(boxes: Tensor, scores: Tensor, iou_th: float) -> Tensor:
    _assert_has_ops()
    return torch.ops.torchvision.nms(boxes, scores, iou_th)


def decode(regressions, anchors, weights=(1., 1., 1., 1.)):
    # type: (Tensor, Tensor, Tuple[int]) -> Tensor
    dtype, device = regressions.dtype, regressions.device
    anchors = anchors.to(regressions.dtype)
    weights = torch.as_tensor(weights, dtype=dtype, device=device)

    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    cx = (anchors[:, 2] + anchors[:, 0])/2
    cy = (anchors[:, 3] + anchors[:, 1])/2

    wx, wy, ww, wh = weights
    dx = regressions[:, 0::4] / wx
    dy = regressions[:, 1::4] / wy
    dw = regressions[:, 2::4] / ww
    dh = regressions[:, 3::4] / wh

    # Prevent sending too large values into torch.exp()
    dw = torch.clamp(dw, max=math.log(1000./16))
    dh = torch.clamp(dh, max=math.log(1000./16))

    pred_cx = dx * widths[:, None] + cx[:, None]
    pred_cy = dy * heights[:, None] + cy[:, None]
    pred_w = torch.exp(dw) * widths[:, None]
    pred_h = torch.exp(dh) * heights[:, None]

    bbx1 = pred_cx - pred_w/2
    bbx2 = pred_cy - pred_h/2
    bbx3 = pred_cx + pred_w/2
    bbx4 = pred_cy + pred_h/2
    boxes = torch.stack((bbx1, bbx2, bbx3, bbx4), dim=2).flatten(1)
    return boxes


def clip_box(boxes: Tensor, size: Tuple[int, int]) -> Tensor:
    dim = boxes.dim()
    boxes_x = boxes[..., 0::2]
    boxes_y = boxes[..., 1::2]
    h, w = size

    boxes_x = boxes_x.clamp(min=0, max=w)
    boxes_y = boxes_y.clamp(min=0, max=h)

    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    return clipped_boxes.reshape(boxes.shape)


def remove_small_boxes(boxes: Tensor, min_size: float) -> Tensor:
    ws, hs = boxes[:, 2]-boxes[:, 0], boxes[:, 3]-boxes[:, 1]
    keep = (ws >= min_size) & (hs >= min_size)
    keep = torch.where(keep)[0]
    return keep


@torch.no_grad()
def post_process(logits, regressions, anchors, image_shape, score_th=0.05, nms_th=0.5, detections_per_img=300):
    # type: (Tensor, Tensor, Tensor, Tuple[int, int], float, float, int) -> List[Tensor]
    device = logits.device
    num_classses = logits.shape[-1]
    scores = torch.sigmoid(logits)
    labels = torch.arange(num_classses, device=device)
    labels = labels.view(1, -1).expand_as(scores)

    boxes = decode(regressions, anchors)
    boxes = clip_box(boxes, image_shape)

    image_boxes = []
    image_scores = []
    image_labels = []
    for class_index in range(1, num_classses):
        # remove low scoring boxes
        idxs = torch.gt(scores[:, class_index], score_th)
        boxes_per_class, scores_per_class, labels_per_class = \
            boxes[idxs], scores[idxs, class_index], labels[idxs, class_index]

        # remove empty boxes
        keep = remove_small_boxes(boxes_per_class, min_size=1e-2)
        boxes_per_class, scores_per_class, labels_per_class = \
            boxes_per_class[keep], scores_per_class[keep], labels_per_class[keep]

        keep = nms(boxes_per_class, scores_per_class, nms_th)
        keep = keep[:detections_per_img]
        boxes_per_class, scores_per_class, labels_per_class = \
            boxes_per_class[keep], scores_per_class[keep], labels_per_class[keep]

        image_boxes.append(boxes_per_class)
        image_scores.append(scores_per_class)
        image_labels.append(labels_per_class)

    boxes = torch.cat(image_boxes, dim=0)
    scores = torch.cat(image_scores, dim=0)
    labels = torch.cat(image_labels, dim=0)

    return boxes, scores, labels


class AnchorGenerator(nn.Module):
    def __init__(
        self,
        sizes=((128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),),
    ):
        super().__init__()
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}

    def forward(self, inputs, feature_maps):
        # type: (Tensor, List[Tensor]) -> List[Tensor]
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        image_size = inputs.shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        strides = [
            [torch.tensor(image_size[0] // g[0], dtype=torch.int64, device=device),
             torch.tensor(image_size[1] // g[1], dtype=torch.int64, device=device)]
            for g in grid_sizes
        ]

        self.set_cell_anchors(dtype, device)
        anchors_over_all_feat_maps = self.cached_grid_anchors(
            grid_sizes, strides)

        anchors = jit.annotate(List[List[Tensor]], [])
        for _ in range(inputs.shape[0]):
            anchors_in_image = []
            for anchors_per_feat_map in anchors_over_all_feat_maps:
                anchors_in_image.append(anchors_per_feat_map)
            anchors.append(anchors_in_image)
        anchors = [torch.cat(anchors_per_image)
                   for anchors_per_image in anchors]
        self._cache.clear()
        return anchors

    def set_cell_anchors(self, dtype, device):
        if self.cell_anchors is not None:
            if self.cell_anchors[0].device == device:
                return

        cell_anchors = []
        for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios):
            anchor = self.generate_anchors(sizes, aspect_ratios, dtype, device)
            cell_anchors.append(anchor)
        self.cell_anchors = cell_anchors

    def generate_anchors(self, scales, aspect_ratios, dtype=torch.float32, device="cpu"):
        # type: (List[int], List[float], int, str) -> Tensor
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(
            aspect_ratios, dtype=dtype, device=device)
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios

        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)

        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1)/2
        return base_anchors.round()

    def cached_grid_anchors(self, grid_sizes, strides):
        # type: (List[List[int]], List[List[Tensor]]) -> List[Tensor]
        key = str(grid_sizes) + str(strides)
        if key in self._cache:
            return self._cache[key]
        anchors = self.grid_anchors(grid_sizes, strides)
        self._cache[key] = anchors
        return anchors

    def grid_anchors(self, grid_sizes, strides):
        # type: (List[List[int]], List[List[Tensor]]) -> List[Tensor]
        '''在featuremap的每一个像素上生成anchor'''
        anchors = []
        cell_anchors = self.cell_anchors
        assert cell_anchors is not None
        # 不同尺度的feature_map的个数要与AnchorGenerator的sizes的个数一致
        assert len(grid_sizes) == len(strides) == len(cell_anchors)

        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            grid_h, grid_w = size
            stride_h, stride_w = stride
            device = base_anchors.device

            shifts_x = torch.arange(0, grid_w) * stride_w
            shifts_y = torch.arange(0, grid_h) * stride_h
            shifts_x.to(device, torch.float32)
            shifts_y.to(device, torch.float32)

            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            anchors.append(
                (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4)
            )

        return anchors


class Matcher(object):

    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    def __init__(self, high_th, low_th, allow_low_quality_matches=False) -> None:
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        assert low_th <= high_th
        self.high_th = high_th
        self.low_th = low_th
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, iou_matrix: torch.Tensor):
        # iou_matrix is M(gt) x N(predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        # values: best iou score, indices: gt box index
        values, indices = iou_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = indices.clone()
        else:
            all_matches = None

        # Assign candidate matches with low quality to negative (unassigned) values
        below_low_th = values < self.low_th
        between_th = (values >= self.low_th) & (values < self.high_th)

        indices[below_low_th] = self.BELOW_LOW_THRESHOLD
        indices[between_th] = self.BETWEEN_THRESHOLDS

        if self.allow_low_quality_matches:
            assert all_matches is not None
            self.set_low_quality_matches_(indices, all_matches, iou_matrix)

        return indices

    def set_low_quality_matches_(self, indices, all_matches, iou_matrix):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> None
        values, _ = iou_matrix.max(1)
        gt_pred_pairs_of_highest_score = torch.where(
            iou_matrix == values[:, None]
        )

        pred_idx_to_update = gt_pred_pairs_of_highest_score[1]
        indices[pred_idx_to_update] = all_matches[pred_idx_to_update]
