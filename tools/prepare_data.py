import sys
import os
import os.path as osp
BASE_DIR = osp.abspath(osp.dirname('../'))
sys.path.append(BASE_DIR)
from utils.regressor import retail_dataset
from utils.regressor.config import A_PIC_ROOT, A_JSON_FILE, B_PIC_ROOT, B_JSON_FILE,\
    TRAIN_SAVE_DIR, TEST_PIC_ROOT, TEST_JSON_FILE, TEST_SAVE_DIR, PAIR_PATH, \
    SIM_RATIO, TOTAL_PAIR, INTERVAL

# # train data prepare
# A_PIC_ROOT = osp.join(BASE_DIR, 'data/train/a_images')
# A_JSON_FILE = osp.join(BASE_DIR, 'data/train/a_annotations.json')
#
# B_PIC_ROOT = osp.join(BASE_DIR, 'data/train/b_images')
# B_JSON_FILE = osp.join(BASE_DIR, 'data/train/b_annotations.json')
#
# TRAIN_SAVE_DIR = osp.join(BASE_DIR, 'data/cropped_train')
#
# # test data prepare
# TEST_PIC_ROOT = osp.join(BASE_DIR, 'data/test/b_images')
# TEST_JSON_FILE = osp.join(BASE_DIR, 'data/test/b_annotations.json')
#
# TEST_SAVE_DIR = osp.join(BASE_DIR, 'data/cropped_test')
# PAIR_PATH = osp.join(BASE_DIR, 'data/pair.txt')
#
# SIM_RATIO = 0.5
# TOTAL_PAIR = 10000

# retail_dataset.crop_images(A_PIC_ROOT, A_JSON_FILE, TRAIN_SAVE_DIR, suffix='a')
# retail_dataset.crop_images(B_PIC_ROOT, B_JSON_FILE, TRAIN_SAVE_DIR, suffix='b')

# retail_dataset.crop_images(TEST_PIC_ROOT, TEST_JSON_FILE, TEST_SAVE_DIR, suffix='b')

retail_dataset.gen_pair(TEST_SAVE_DIR, save_path=PAIR_PATH, 
                        sim_ratio=SIM_RATIO, total_num=TOTAL_PAIR, 
                        interval=INTERVAL)

