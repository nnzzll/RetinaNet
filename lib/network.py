import math
import torch
import torch.nn as nn
from typing import List


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv2(self.conv1(x))


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_filters=[32, 64, 128, 256]):
        super().__init__()
        self.enc1 = Block(in_channels, n_filters[0])
        self.enc2 = Block(n_filters[0], n_filters[1])
        self.enc3 = Block(n_filters[1], n_filters[2])
        self.enc4 = Block(n_filters[2], n_filters[3])
        self.pool = nn.MaxPool2d(2, 2)
        self.out = nn.Conv2d(n_filters[3], out_channels, 1)

    def forward(self, x):
        features = []
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        features.append(self.out(e4))
        return features


class Head(nn.Module):
    def __init__(self, in_channels, n_classes, n_anchors):
        super().__init__()
        self.classification_head = ClassificationHead(
            in_channels, n_classes, n_anchors)
        self.regression_head = RegressionHead(in_channels, n_anchors)

    def forward(self, x):
        return self.classification_head(x), self.regression_head(x)


class ClassificationHead(nn.Module):
    def __init__(self, in_channels, n_classes, n_anchors, prior_probability=0.01):
        super().__init__()
        self.n_classes = n_classes
        conv_list = []
        for _ in range(4):
            conv_list.append(nn.Conv2d(in_channels, in_channels, 3, 1, 1))
            conv_list.append(nn.ReLU())
        self.conv = nn.Sequential(*conv_list)
        self.cls_logits = nn.Conv2d(in_channels, n_classes*n_anchors, 3, 1, 1)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)
        nn.init.normal_(self.cls_logits.weight, std=0.01)
        nn.init.constant_(self.cls_logits.bias,
                          -math.log((1 - prior_probability) / prior_probability))

    def forward(self, x):
        # type: (List[torch.Tensor]) -> torch.Tensor
        all_cls_logits = []

        for features in x:
            cls_logits = self.conv(features)
            cls_logits = self.cls_logits(cls_logits)
            # Permute classification output from (N, A * K, H, W) to (N, HWA, K).
            N, _, H, W = cls_logits.shape
            cls_logits = cls_logits.view(N, -1, self.n_classes, H, W)
            cls_logits = cls_logits.permute(0, 3, 4, 1, 2)
            cls_logits = cls_logits.reshape(N, -1, self.n_classes)

            all_cls_logits.append(cls_logits)

        return torch.cat(all_cls_logits, dim=1)


class RegressionHead(nn.Module):
    def __init__(self, in_channels, n_anchors):
        super().__init__()

        conv_list = []
        for _ in range(4):
            conv_list.append(nn.Conv2d(in_channels, in_channels, 3, 1, 1))
            conv_list.append(nn.ReLU())
        self.conv = nn.Sequential(*conv_list)
        self.bbox_reg = nn.Conv2d(in_channels, 4*n_anchors, 3, 1, 1)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)
        nn.init.normal_(self.bbox_reg.weight, std=0.01)
        nn.init.zeros_(self.bbox_reg.bias)

    def forward(self, x):
        # type: (List[torch.Tensor]) -> torch.Tensor
        all_bbox_regression = []

        for features in x:
            bbox_regression = self.conv(features)
            bbox_regression = self.bbox_reg(bbox_regression)

            # Permute bbox regression output from (N, 4 * A, H, W) to (N, HWA, 4)
            N, _, H, W = bbox_regression.shape
            bbox_regression = bbox_regression.view(N, -1, 4, H, W)
            bbox_regression = bbox_regression.permute(0, 3, 4, 1, 2)
            bbox_regression = bbox_regression.reshape(N, -1, 4)

            all_bbox_regression.append(bbox_regression)

        return torch.cat(all_bbox_regression, dim=1)


class RetinaNet(nn.Module):
    def __init__(self, in_channels, n_classes, n_anchors):
        super().__init__()
        self.backbone = UNet(in_channels, 256)
        self.head = Head(256, n_classes, n_anchors)

    def forward(self, x):
        features = self.backbone(x)
        cls_logits, bbox_regression = self.head(features)
        return features, cls_logits, bbox_regression


if __name__ == '__main__':
    model = RetinaNet(1, 2, 9).cuda()
    inputs = torch.rand((1, 1, 128, 320)).cuda()
    features, cls_logits, bbox_regression = model(inputs)
    print(cls_logits.shape)
    print(bbox_regression.shape)
