import torch
from torch import Tensor
from torch.nn import functional as F
from typing import List, Dict, Tuple


def _sum(x: List[Tensor]) -> Tensor:
    '''对列表中的Tensor进行求和,猜测如果使用Python内置的sum可能无法反向传播'''
    res = x[0]
    for i in x[1:]:
        res += i
    return res


def compute_cls_loss(targets, logits, indices, BETWEEN_THRESHOLDS=-2):
    # type: (List[Dict[str, Tensor]], Tensor, List[Tensor], int) -> Tensor
    # shape: (_, [B, N, 2], [N])
    losses = []

    for target, logit, index in zip(targets, logits, indices):
        foreground_indices = index >= 0
        num_foreground = foreground_indices.sum()

        # create the target classification
        gt_classes_target = torch.zeros_like(logit)  # [N, 2]
        gt_classes_target[
            foreground_indices,
            target['labels'][index[foreground_indices]]
        ] = 1.0

        # find indices for which anchors should be ignored
        valid_indices = index != BETWEEN_THRESHOLDS
        losses.append(
            sigmoid_focal_loss(
                logit[valid_indices],
                gt_classes_target[valid_indices],
                reduction='sum',
            ) / max(1, num_foreground)
        )

    return sum(losses) / max(1, len(targets))


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
):
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def compute_bbx_loss(targets, bbox_regression, anchors, indices):
    # type: (List[Dict[str, Tensor]], Tensor, List[Tensor], List[Tensor]) -> Tensor
    losses = []

    for target, bbox, anchor, index in zip(targets, bbox_regression, anchors, indices):
        foreground_indices, = torch.where(index >= 0)
        num_foreground = foreground_indices.numel()

        matched_gt_boxes = target["boxes"][index[foreground_indices]]
        bbox = bbox[foreground_indices, :]
        anchor = anchor[foreground_indices, :]

        target_regression = encode_boxes(matched_gt_boxes, anchor)

        losses.append(
            F.l1_loss(
                bbox,
                target_regression,
                reduction='sum'
            ) / max(1, num_foreground)
        )

    return sum(losses) / max(1, len(targets))


def encode_boxes(gt_boxes: Tensor, pr_boxes, weights=(1., 1., 1., 1.)):
    dtype = gt_boxes.dtype
    device = gt_boxes.device
    weights = torch.as_tensor(weights, dtype=dtype, device=device)
    return encode(gt_boxes, pr_boxes, weights)


@torch.jit._script_if_tracing
def encode(gt_boxes, pr_boxes, weights):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    wx, wy, ww, wh = weights

    pr_x1 = pr_boxes[:, 0].unsqueeze(1)
    pr_y1 = pr_boxes[:, 1].unsqueeze(1)
    pr_x2 = pr_boxes[:, 2].unsqueeze(1)
    pr_y2 = pr_boxes[:, 3].unsqueeze(1)

    gt_x1 = gt_boxes[:, 0].unsqueeze(1)
    gt_y1 = gt_boxes[:, 1].unsqueeze(1)
    gt_x2 = gt_boxes[:, 2].unsqueeze(1)
    gt_y2 = gt_boxes[:, 3].unsqueeze(1)

    pr_w = pr_x2 - pr_x1
    pr_h = pr_y2 - pr_y1
    pr_x = (pr_x1 + pr_x2) / 2
    pr_y = (pr_y1 + pr_y2) / 2

    gt_w = gt_x2 - gt_x1
    gt_h = gt_y2 - gt_y1
    gt_x = (gt_x1 + gt_x2) / 2
    gt_y = (gt_y1 + gt_y2) / 2

    dx = wx * (gt_x - pr_x) / pr_w
    dy = wy * (gt_y - pr_y) / pr_h
    dw = ww * torch.log(gt_w / pr_w)
    dh = wh * torch.log(gt_h / pr_h)

    targets = torch.cat((dx, dy, dw, dh), dim=1)
    return targets
