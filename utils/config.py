import os.path as osp
import os
BASE_DIR = osp.abspath(osp.join(osp.dirname(__file__), osp.pardir))
# BASE_DIR = osp.abspath('.')

"""Training and testing configs"""
BATCH_SIZE = 256
SAVE_FREQ = 1
TEST_FREQ = 1
TOTAL_EPOCH = 300

RESUME = 0
SAVE_DIR = './arcface_runs'
MODEL_PRE = 'Retail_'

GPU = 0

"""regressor"""
# train data prepare, crop images
A_PIC_ROOT = osp.join(BASE_DIR, 'data/train/a_images')
A_JSON_FILE = osp.join(BASE_DIR, 'data/train/a_annotations.json')

B_PIC_ROOT = osp.join(BASE_DIR, 'data/train/b_images')
B_JSON_FILE = osp.join(BASE_DIR, 'data/train/b_annotations.json')

TRAIN_SAVE_DIR = osp.join(BASE_DIR, 'data/cropped_train')

# test data prepare, crop images
TEST_PIC_ROOT = osp.join(BASE_DIR, 'data/test/b_images')
TEST_JSON_FILE = osp.join(BASE_DIR, 'data/test/b_annotations.json')

TEST_SAVE_DIR = osp.join(BASE_DIR, 'data/cropped_test')
PAIR_PATH = osp.join(BASE_DIR, 'data/pair.txt')

# test pair config
SIM_RATIO = 0.5  
TOTAL_PAIR = 10000
INTERVAL = 500


"""detector"""
DETECTOR_TRAIN_DATA_DIR = osp.join(BASE_DIR, 'data/retail/train')
DETECTOR_VAL_DATA_DIR = osp.join(BASE_DIR, 'data/retail/val')
