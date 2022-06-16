import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot(img, pred_boxes, scores=None, gt_boxes=None, title=None, filename='result.png'):
    plt.cla()
    ax = plt.gca()
    ax.imshow(img)
    for i in range(len(pred_boxes)):
        xmin, ymin, xmax, ymax = pred_boxes[i]
        pred = patches.Rectangle(
            (xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, facecolor='none', edgecolor='r')
        ax.add_patch(pred)
        if scores:
            ax.text(xmin,ymin,f"{scores[i]:.2f}",color='white')

    if gt_boxes:
        for boxes in gt_boxes:
            xmin, ymin, xmax, ymax = boxes
            gt = patches.Rectangle(
                (xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, facecolor='none', edgecolor='g')
            ax.add_patch(gt)

    if title:
        ax.set_title(title)
    plt.axis("off")
    plt.savefig(filename)
    plt.show()