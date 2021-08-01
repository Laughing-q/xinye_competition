import os.path as osp

BASE_DIR = osp.abspath(osp.dirname(__file__))
import sys

sys.path.insert(0, BASE_DIR)
from model.detector import Yolov5
from utils.general import xyxy2xywh
from utils.regressor.distance_calculation_arcface import multi_matching
from model.regressor.create_regressor import create_model, Regressor, Ensemble, \
    load_regressor
from utils.regressor import retail_eval
from utils.plots import plot_one_box
from utils.config import IMAGE_RESOLUTION, CONCAT, MEAN, NUM_WORKERS
import torch
import cv2
import random
import glob
import json
import tqdm
import os
import timm
import scipy.io

random.seed(0)

REGRESS_THRES = 0.0
DETECT_THRES = 0.001
IOU_THRES = 0.5
REGRESS_INPUT_SIZE = (IMAGE_RESOLUTION, IMAGE_RESOLUTION)  # (w, h)
DETECT_MODE = 'x'
REGRESS_BATCH_SIZE = 16
COLORS = [[random.randint(0, 255) for _ in range(3)]
          for _ in range(116)]

# DETECTOR_WEIGHT_PATH = osp.join(BASE_DIR, 'model_files/yolov5x_best.pth')
# DETECTOR_CFG_PATH = osp.join(BASE_DIR, 'model/yolov5x_best.yaml')
# DETECTOR_WEIGHT_PATH = osp.join(BASE_DIR, 'model_files/yolov5x_single.pth')
# DETECTOR_CFG_PATH = osp.join(BASE_DIR, 'model/yolov5x_single.yaml')
DETECTOR_WEIGHT_PATH = osp.join(BASE_DIR, 'model_files/yolov5x_RPC.pth')
DETECTOR_CFG_PATH = osp.join(BASE_DIR, 'model/yolov5x_RPC.yaml')
# DETECTOR_WEIGHT_PATH = osp.join(BASE_DIR, 'model_files/yolov5x_RPC_all.pth')
# DETECTOR_CFG_PATH = osp.join(BASE_DIR, 'model/yolov5x_RPC_all.yaml')

# REGRESS_WEIGHT_PATH = osp.join(BASE_DIR, 'model_files/swin_large_cgd_epoch126_0.9993_1024.ckpt')
# REGRESS_WEIGHT_PATH = osp.join(BASE_DIR, 'model_files/siwn_large_cgd_epoch049_99.99.ckpt')
REGRESS_WEIGHT_PATH = osp.join(BASE_DIR, 'model_files/swinlarge_arcfade_epoch126_0.9966_1024.ckpt')
# REGRESS_WEIGHT_PATH_1 = osp.join(BASE_DIR, 'model_files/swin_large_028epoch_99.97_0.3506.ckpt')
REGRESS_WEIGHT_PATH_1 = osp.join(BASE_DIR, 'model_files/swin_large_cgd_epoch126_0.9993_1024.ckpt')
REGRESS_WEIGHT_PATH_2 = osp.join(BASE_DIR, 'model_files/swin_small_cgd_epoch040._9.9633.ckpt')
RESULT_SAVE_PATH = osp.join(BASE_DIR, 'submit/output.json')

TEST_IMAGES_PATH = osp.join(BASE_DIR, 'data/test/a_images')
TEST_JSON_PATH = osp.join(BASE_DIR, 'data/test/a_annotations.json')
# TEST_IMAGES_PATH = "/d/competition/retail/Preliminaries/test/a_images"
# TEST_JSON_PATH = '/d/competition/retail/Preliminaries/test/a_annotations.json'

RETRIEVAL_IMAGE_PATH = osp.join(BASE_DIR, 'data/test/b_images')
RETRIEVAL_JSON_PATH = osp.join(BASE_DIR, 'data/test/b_annotations.json')

# PIC_SAVE_PATH = './submit/normal+min'
# os.makedirs(PIC_SAVE_PATH, exist_ok=True)

def run():
    # total_boxes = 0
    detector = Yolov5(weight_path=DETECTOR_WEIGHT_PATH,
                      cfg=DETECTOR_CFG_PATH,
                      device='0', img_hw=(640, 640))
    detector.show = False
    print('load detector successfully!')

    # create model
    regressor = load_regressor(weights=[REGRESS_WEIGHT_PATH, REGRESS_WEIGHT_PATH_1, REGRESS_WEIGHT_PATH_2],
                               model_names=['swin_transformer', 'swin_transformer', 'swin_transformer'],
                               pretrained=[False, False, False],
                               cgd=[True, True, True],
                               swin_type=['large', 'large', 'small'],
                               class_num=[111, 111, 116],
                               feature_dim=[1024, 1024, 512],
                               concat=[True, True, True])
    # regressor = load_regressor(weights=[REGRESS_WEIGHT_PATH],
    #                            model_names=['swin_transformer'],
    #                            pretrained=[False],
    #                            cgd=[True],
    #                            swin_type=['large'],
    #                            class_num=[107],
    #                            feature_dim=[1024],
    #                            concat=[True])
    print('load regressor successfully!')

    test_dataset = retail_eval.RetailDataset(pic_root=RETRIEVAL_IMAGE_PATH,
                                             json_file=RETRIEVAL_JSON_PATH,
                                             img_size=IMAGE_RESOLUTION)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=REGRESS_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=False)
    mat = regressor.generateBase(test_loader=test_loader)

    database = mat['feature']
    # database_ = mat_['feature']
    # database__ = mat__['feature']
    category_base = mat['class']

    with open(TEST_JSON_PATH, 'r') as f:
        test_json = json.load(f)
    # print(test_json.keys())

    # get map file_name -> image_id
    map_list = {}
    for img_info in test_json['images']:
        map_list[img_info['file_name']] = img_info['id']

    annotation = []

    img_paths = glob.glob(osp.join(TEST_IMAGES_PATH, '*'))
    pbar = tqdm.tqdm(img_paths, total=len(img_paths))
    for path in pbar:
        pbar.desc = f"{path}"
        # pbar.set_description(f"{path}")
        img = cv2.imread(path)
        file_name = osp.basename(path)

        img, img_raw = detector.preprocess(img, auto=True)
        preds = detector.dynamic_detect(img, [img_raw],
                                        conf_threshold=DETECT_THRES,
                                        iou_threshold=IOU_THRES,
                                        agnostic=True,
                                        wbf=False)
        # total_boxes += len(preds[0])
        # cv2.putText(img_raw, f"{len(preds[0])}", (10, 60), 0, 2, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
        for det in preds:
            if det is None or len(det) == 0:
                continue
            total_goods = []
            for *xyxy, conf, _ in det:
                x1, y1, x2, y2 = [int(temp) for temp in xyxy]
                goods = img_raw[y1:y2, x1:x2, :]
                goods = cv2.cvtColor(goods, cv2.COLOR_BGR2RGB)
                total_goods.append(cv2.resize(goods, REGRESS_INPUT_SIZE))

            features = regressor(total_goods, batch_size=REGRESS_BATCH_SIZE)
            categories, scores = regressor.matching(database=database,
                                                    category=category_base,
                                                    features=features)
            categories, scores = regressor.selectResutlt(categories=categories,
                                                        scores=scores)

            detect_confs = det[:, 4]
            det_boxes = det[:, :4]
            boxes = xyxy2xywh(det_boxes)
            boxes[:, :2] -= boxes[:, 2:] / 2
            for i in range(len(boxes)):
                if scores[i] > REGRESS_THRES:
                    annotation.append({'image_id': map_list[file_name],
                                       'category_id': int(categories[i]),
                                       'bbox': [round(x, 3) for x in boxes[i].tolist()],
                                       'score': round(float(detect_confs[i] * scores[i]), 5)})
                    label = '%s' % (int(categories[i]))
                    label = f'{int(categories[i])} {float(detect_confs[i] * scores[i]):.2f}'
                    # plot_one_box(det_boxes[i], img_raw, label=label,
                    #              color=COLORS[int(categories[i])], 
                    #              line_thickness=2)
        # cv2.imwrite(osp.join(PIC_SAVE_PATH, file_name), img_raw)
    test_json['annotations'] = annotation
    with open(RESULT_SAVE_PATH, 'w') as fw:
        json.dump(test_json, fw, indent=4)
    # print(total_boxes)
"""
torch_nms+0.5: 52199
torchvision+0.5: 52579, 50805
torch_nms+Min+0.5: 47892
torch_nms+Min+0.85: 51157
torchvision+0.5+Min+0.85: 49424

yolov5x_RPC_all: 52440
"""

if __name__ == "__main__":
    # print(osp.join(BASE_DIR, '*'))
    run()
