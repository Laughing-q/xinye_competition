import sys
import os
import os.path as osp
BASE_DIR = osp.abspath(osp.join(osp.dirname(__file__), osp.pardir))
sys.path.append(BASE_DIR)
from utils.regressor import retail_dataset
from utils.config import A_PIC_ROOT, A_JSON_FILE, B_PIC_ROOT, B_JSON_FILE,\
    TRAIN_SAVE_DIR, TEST_PIC_ROOT, TEST_JSON_FILE, TEST_SAVE_DIR, PAIR_PATH, \
    SIM_RATIO, TOTAL_PAIR, INTERVAL, DETECTOR_TRAIN_DATA_DIR, DETECTOR_VAL_DATA_DIR,\
    DETECTOR_TEST_DATA_DIR
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--regressor-data', action='store_true', help='prepare regressor data')
parser.add_argument('--detector-data', action='store_true', help='prepare detector data')
parser.add_argument('--generate-pair', action='store_true', help='generate pair')

opt = parser.parse_args()
# for regressor
if opt.regressor_data:
    print('regressor')
    retail_dataset.crop_images(A_PIC_ROOT, A_JSON_FILE, TRAIN_SAVE_DIR, suffix='a')
    retail_dataset.crop_images(B_PIC_ROOT, B_JSON_FILE, TRAIN_SAVE_DIR, suffix='b')

    retail_dataset.crop_images(TEST_PIC_ROOT, TEST_JSON_FILE, TEST_SAVE_DIR, suffix='b')


if opt.generate_pair:
    print('generate pair.txt')
    retail_dataset.gen_pair(TEST_SAVE_DIR, save_path=PAIR_PATH, 
                            sim_ratio=SIM_RATIO, total_num=TOTAL_PAIR, 
                            interval=INTERVAL)


# for detector
if opt.detector_data:
    print('detector')
    script_path = osp.join(BASE_DIR, 'utils/coco2yolo.py')
    cmd_train = f"python {script_path} --json-file {A_JSON_FILE} --images-source {A_PIC_ROOT} --save-dir {DETECTOR_TRAIN_DATA_DIR} --single-cls"
    cmd_val = f"python {script_path} --json-file {B_JSON_FILE} --images-source {B_PIC_ROOT} --save-dir {DETECTOR_VAL_DATA_DIR} --single-cls"
    cmd_test = f"python {script_path} --json-file {TEST_JSON_FILE} --images-source {TEST_PIC_ROOT} --save-dir {DETECTOR_TEST_DATA_DIR} --single-cls"
    subprocess.call(cmd_train, shell=True)
    subprocess.call(cmd_val, shell=True)
    subprocess.call(cmd_test, shell=True)
