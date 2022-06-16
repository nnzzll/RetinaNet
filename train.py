import torch
import numpy as np
import torch.optim as optim

from PIL import Image
from lib.network import RetinaNet
from lib.ops import AnchorGenerator, Matcher, compute_iou
from lib.loss import compute_bbx_loss, compute_cls_loss


if __name__ == '__main__':
    BEST_SCORE = 1e8
    img = np.array(Image.open("007.png"))
    Y, X = np.where(img == 11)
    boxes = []
    boxes.append([X.min(), Y.min(), X.max(), Y.max()])
    inputs = np.expand_dims(img, 0).astype(np.float32)
    inputs = torch.Tensor(inputs).unsqueeze(0)

    targets = {}
    targets["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
    targets["labels"] = torch.as_tensor([1], dtype=torch.int64)
    targets = [targets]

    model = RetinaNet(1, 2, 9)
    anchor_generator = AnchorGenerator(((16, 20, 25),))
    matcher = Matcher(0.5, 0.4, True)
    optimizer = optim.SGD(model.parameters(), 1e-3, 0.9, weight_decay=5e-4)

    for i in range(100):
        features, cls_logits, bbox_regression = model(inputs)
        anchors = anchor_generator(inputs, features)

        matched_idxs = []
        for anchor, target in zip(anchors, targets):
            iou_matrix = compute_iou(target["boxes"], anchor)
            idxs = matcher(iou_matrix)
            matched_idxs.append(idxs)

        cls_loss = compute_cls_loss(targets, cls_logits, matched_idxs)
        bbx_loss = compute_bbx_loss(targets, bbox_regression, anchors, matched_idxs)
        loss = cls_loss + bbx_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Ep:{i+1}\tcls loss:{cls_loss.item():.6f}\tbbox_loss:{bbx_loss.item()}")
        if (i+1)%10==0:
            if cls_loss.item() + bbx_loss.item() < BEST_SCORE:
                BEST_SCORE = cls_loss.item() + bbx_loss.item()
                torch.save(model.state_dict(), f"weights.pth")