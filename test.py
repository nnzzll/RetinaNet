import torch
import numpy as np
from PIL import Image
from lib.network import RetinaNet
from lib.ops import AnchorGenerator, post_process
from lib.utils import plot


if __name__ == '__main__':
    img = np.array(Image.open("007.png"))
    Y, X = np.where(img == 11)
    boxes = []
    boxes.append([X.min(), Y.min(), X.max(), Y.max()])
    inputs = np.expand_dims(img, 0).astype(np.float32)  # [1, H, W]
    inputs = torch.Tensor(inputs).unsqueeze(0)  # [1, 1, H, W]

    targets = {}
    targets["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
    targets["labels"] = torch.as_tensor([1], dtype=torch.int64)

    model = RetinaNet(1, 2, 9)
    anchor_generator = AnchorGenerator(((16, 20, 25),))
    weights = torch.load("weights.pth")
    model.load_state_dict(weights)

    with torch.no_grad():
        features, cls_logits, bbox_regression = model(inputs)
        anchors = anchor_generator(inputs, features)
        boxes, scores, labels = post_process(
            cls_logits[0], bbox_regression[0], anchors[0], img.shape)
        print(boxes)
        print(scores)
        print(labels)
    plot(img, list(boxes), list(scores), list(targets["boxes"]))