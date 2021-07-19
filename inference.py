import os.path as osp
BASE_DIR = osp.abspath(osp.dirname(__file__))
import sys
sys.path.insert(0, BASE_DIR)
from model.detector import Yolov5
from model.swin_transformer import SwinTransformer
from utils.general import xyxy2xywh
from utils.regressor.distance_calculation_arcface import multi_matching
from utils.regressor import retail_eval
from utils.plots import plot_one_box
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


REGRESS_THRES = 0.276
DETECT_THRES = 0.001
IOU_THRES = 0.4
REGRESS_INPUT_SIZE = (112, 112)  # (w, h)
DETECT_MODE = 'x'
REGRESS_BATCH_SIZE = 32
COLORS = [[random.randint(0, 255) for _ in range(3)]
               for _ in range(116)]

DETECTOR_WEIGHT_PATH = osp.join(BASE_DIR, 'model_files/yolov5x_best.pth')
DETECTOR_CFG_PATH = osp.join(BASE_DIR, 'model/yolov5x_best.yaml')

# REGRESS_WEIGHT_PATH = osp.join(BASE_DIR, 'model_files/efficientnetb4_99.95_0.592392.ckpt')
REGRESS_WEIGHT_PATH = osp.join(BASE_DIR, 'model_files/swintransformer+circleloss_99.9792_0.2760.ckpt')
RESULT_SAVE_PATH = osp.join(BASE_DIR, 'submit/output.json')


TEST_IMAGES_PATH = osp.join(BASE_DIR, 'data/test/a_images')
TEST_JSON_PATH = osp.join(BASE_DIR, 'data/test/a_annotations.json')
# TEST_IMAGES_PATH = "/d/competition/retail/Preliminaries/test/a_images"
# TEST_JSON_PATH = '/d/competition/retail/Preliminaries/test/a_annotations.json'

RETRIEVAL_IMAGE_PATH = osp.join(BASE_DIR, 'data/test/b_images')
RETRIEVAL_JSON_PATH = osp.join(BASE_DIR, 'data/test/b_annotations.json')
# RETRIEVAL_IMAGE_PATH = "/d/competition/retail/Preliminaries/test/b_images"
# RETRIEVAL_JSON_PATH = '/d/competition/retail/Preliminaries/test/b_annotations.json'


def run():
    detector = Yolov5(weight_path=DETECTOR_WEIGHT_PATH, 
                      cfg=DETECTOR_CFG_PATH,
                      device='0', img_hw=(640, 640))
    detector.show = False
    
    # efficientnet
    # regressor = timm.create_model('efficientnet_b4', pretrained=False, num_classes=256).cuda()
    # regressor.load_state_dict(torch.load(REGRESS_WEIGHT_PATH)['net_state_dict'])
    # regressor.eval()

    # swin transformer
    regressor = SwinTransformer(img_size=112, num_classes=256).cuda()
    regressor.load_state_dict(torch.load(REGRESS_WEIGHT_PATH)['net_state_dict'])
    regressor.eval()


    test_dataset = retail_eval.RetailDataset(pic_root=RETRIEVAL_IMAGE_PATH, 
                                            json_file=RETRIEVAL_JSON_PATH)

    mat = retail_eval.getFeatureFromTorch(regressor, test_dataset, batch_size=REGRESS_BATCH_SIZE)
    database = mat['feature']
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
    for path in tqdm.tqdm(img_paths, total=len(img_paths)):
        img = cv2.imread(path)
        file_name = osp.basename(path)

        img, img_raw = detector.preprocess(img, auto=True)
        preds = detector.dynamic_detect(img, [img_raw], 
                                        conf_threshold=DETECT_THRES,
                                        iou_threshold=IOU_THRES,
                                        agnostic=True)

        for det in preds:
            if det is None or len(det) == 0:
                continue
            total_goods = []
            for *xyxy, conf, _ in det:
                x1, y1, x2, y2 = [int(temp) for temp in xyxy]
                goods = img_raw[y1:y2, x1:x2, :]
                goods = cv2.cvtColor(goods, cv2.COLOR_BGR2RGB)
                total_goods.append(cv2.resize(goods, REGRESS_INPUT_SIZE))

            categories, scores = multi_matching(img=total_goods, 
                                                database=database, 
                                                category=category_base, 
                                                net=regressor, 
                                                batch_size=REGRESS_BATCH_SIZE)
                                                # batch_size=None)
            detect_confs = det[:, 4]
            det_boxes = det[:, :4]
            boxes = xyxy2xywh(det_boxes)
            boxes[:, :2] -= boxes[:, 2:] / 2
            for i in range(len(boxes)):
                if scores[i] > REGRESS_THRES:
                    annotation.append({'image_id': map_list[file_name],
                                      'category_id':  int(categories[i]),
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

if __name__ == "__main__":
    # print(osp.join(BASE_DIR, '*'))
    run()

