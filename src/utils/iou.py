import numpy as np


def IOU(box1, box2):
    """Compute IoU between detect box1 and gt box2

    Parameters:
    ----------
    box1: numpy array , shape (4, ): x1, y1, x2, y2
        predicted box2
    box2: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth box2

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    # 计算两个bbox的面积
    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[:, 2] - box2[:, 0] + 1) * (box2[:, 3] - box2[:, 1] + 1)
    # 两个bbox交集的面积
    xx1 = np.maximum(box1[0], box2[:, 0])
    yy1 = np.maximum(box1[1], box2[:, 1])
    xx2 = np.minimum(box1[2], box2[:, 2])
    yy2 = np.minimum(box1[3], box2[:, 3])
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    inter = w * h
    # iou=两个bbox交集的面积/(两个bbox的面积之和-两个bbox交集的面积)
    return inter / (area1 + area2 - inter)

