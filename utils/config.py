import os.path as osp
import os

BASE_DIR = osp.abspath(osp.join(osp.dirname(__file__), osp.pardir))
# BASE_DIR = osp.abspath('.')

NUM_THREADS = min(16, os.cpu_count())  # number of multiprocessing threads
NUM_WORKERS = min(16, os.cpu_count())  # number of torch dataloader cpu workers

FEATURE_DIMS = 512
CLASS_NUM = 116

IMAGE_RESOLUTION = 224  # the inference input size, same as training input_size mostly

"""Regressor"""

# Training and testing configs

CONCAT = True  # Whether to concat the prediction results


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
TOTAL_PAIR = 30000
INTERVAL = 3000

SWIN_PRETRAIN = osp.join(BASE_DIR, 'model/regressor/pretrain/swin_small_patch4_window7_224.pth')

# augmentation
AUGMENT_PROBABILITY = {
    'RandomResizedCrop': 0.5,
    'HorizontalFlip': 0.5,
    'VerticalFlip': 0.5,
    'RandomBrightnessContrast': 0.5,
}

"""Detector"""
DETECTOR_TRAIN_DATA_DIR = osp.join(BASE_DIR, 'data/retail/train')
DETECTOR_VAL_DATA_DIR = osp.join(BASE_DIR, 'data/retail/val')
DETECTOR_TEST_DATA_DIR = osp.join(BASE_DIR, 'data/retail/test')

save_args = {
    'num_threads': NUM_THREADS, 
    'num_workers': NUM_WORKERS, 
    'feature_dims': FEATURE_DIMS, 
    'image_resolution': IMAGE_RESOLUTION, 
    'concat': CONCAT, 
    'a_pic_root': A_PIC_ROOT, 
    'a_json_file': A_JSON_FILE, 
    'b_pic_root': B_PIC_ROOT, 
    'b_json_file': B_JSON_FILE, 
    'train_save_dir': TRAIN_SAVE_DIR, 
    'test_pic_root': TEST_PIC_ROOT, 
    'test_json_file': TEST_JSON_FILE, 
    'test_save_dir': TEST_SAVE_DIR, 
    'pair_path': PAIR_PATH,
    'sim_ratio': SIM_RATIO,
    'total_pair': TOTAL_PAIR,
    'interval': INTERVAL,
    'augment': AUGMENT_PROBABILITY,
    'detector_train': DETECTOR_TRAIN_DATA_DIR,
    'detector_val': DETECTOR_VAL_DATA_DIR,
    'detector_test': DETECTOR_TEST_DATA_DIR,
}
