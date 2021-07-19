import logging
import numpy as np
import os
import sys
import zipfile
import json
from itertools import chain
import cv2
import shutil
import time
import argparse
import os.path as osp
from pycocotools.coco import COCO
import tqdm


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]

    # w = box[2] - box[0]
    # h = box[3] - box[1]
    # x = box[0] + w / 2
    # y = box[1] + h / 2
    # x = round(x * dw, 3)
    # w = round(w * dw, 3)
    # y = round(y * dh, 3)
    # h = round(h * dh, 3)
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--json-file',
    type=str,
    default='/d/baidubase/COCO/annotations_trainval2017/annotations/instances_val2017.json',
    help='label json_file')
parser.add_argument(
    '--save-dir',
    type=str,
    default='/d/baidubase/COCO/val_yolo',
    help='save-dir of txt and image')
parser.add_argument('--class-name',
                    nargs='+',
                    type=str,
                    default=[],
                    help='Categories filtered')
parser.add_argument(
    '--intersection',
    action='store_true',
    default=False,
    help='Whether to take the intersection of multiple categories')
parser.add_argument(
    '--images-source',
    type=str,
    # default='/d/baidubase/COCO/train2017',
    default='',
    # parser.add_argument('--images-source', type=str, default='',
    help='copy images from images-source to save-dir/images')
parser.add_argument(
    '--single-cls',
    action='store_true',
    default=False,
    help='Whether use single class')

opt = parser.parse_args()
print(opt)

s_t = time.time()
json_file = opt.json_file  
save_dir = opt.save_dir


ana_txt_save_path = os.path.join(save_dir, "labels")  
img_save_path = os.path.join(save_dir, "images")  

coco = COCO(json_file)

class_name = opt.class_name  # 设置为空，则取全部的类别

intersection = opt.intersection

single_cls = opt.single_cls

catId = coco.getCatIds(catNms=class_name)  # 1~90

if len(class_name):
    if intersection:
        imgIds = coco.getImgIds(catIds=catId)
    else:
        imgIds = [coco.catToImgs[i] for i in catId]
        imgIds = set(chain(*imgIds))
else:
    imgIds = coco.getImgIds()

coco_imgs = coco.loadImgs(imgIds)


if not os.path.exists(ana_txt_save_path):
    os.makedirs(ana_txt_save_path)

if not os.path.exists(img_save_path) and len(opt.images_source):
    os.makedirs(img_save_path)

for img in tqdm.tqdm(coco_imgs, total=len(coco_imgs)):
    filename = img["file_name"]
    img_width = img["width"]
    img_height = img["height"]
    img_id = img["id"]
    if len(opt.images_source):
        shutil.copy(os.path.join(opt.images_source, filename), img_save_path)

    ana_txt_name = osp.basename(osp.splitext(filename)[0]) + ".txt" 
    f_txt = open(os.path.join(ana_txt_save_path, ana_txt_name), 'a')

    for ann in coco.imgToAnns[img_id]:
        if ann['category_id'] in catId:
            box = convert((img_width, img_height), ann["bbox"])
            f_txt.write(
                "%s %s %s %s %s\n" %
                (0 if single_cls else ann["category_id"], box[0], box[1], box[2], box[3]))
    f_txt.close()
    # if opt.mask:
    #     mask_npy = np.stack(mask_npy, axis=0)
    #     print(mask_npy.shape)
    #     np.save(os.path.join(mask_save_path, mask_npy_name), mask_npy)
print(f'spend time:{time.time() - s_t}s')
