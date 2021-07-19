import cv2
import numpy as np
import random
import glob
import tqdm
import os
import torch
import os.path as osp


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(
        x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

img_dir = 'data/retail/total/images'
names = ['retail']
# img_dir = '/d/competition/IFLYTEK/object_detection/华南农业大学_柑橘病虫害识别挑战赛/train/images'
# names = [0, 1, 2]


colors = [[random.randint(0, 255) for _ in range(3)]
          for _ in range(len(names))]

img_ps = glob.glob(osp.join(img_dir, '*'))
cv2.namedWindow('p', cv2.WINDOW_NORMAL)
for img_p in tqdm.tqdm(img_ps, total=len(img_ps)):
    label_p = img_p.replace('images', 'labels')
    label_p = osp.splitext(label_p)[0] + '.txt'
    if not osp.exists(label_p):
        raise FileNotFoundError(f"{label_p} not found.")
    with open(label_p, 'r') as f:
        label = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)  # labels
    if len(label):
        img = cv2.imread(img_p)
        h, w, _ = img.shape
        boxes = label[:, 1:] * [w, h, w, h]
        boxes = xywh2xyxy(boxes)
        cls = label[:, 0]
        for i in range(len(label)):
            plot_one_box(boxes[i], img, label=f'{names[int(cls[i])]}',
                         color=colors[int(cls[i])], line_thickness=2)
        cv2.imshow('p', img)
        if cv2.waitKey(0) == ord('q'):  # q to quit
            break



