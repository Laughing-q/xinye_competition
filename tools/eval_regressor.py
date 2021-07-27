import os.path as osp

BASE_DIR = osp.abspath(osp.join(osp.dirname(__file__), osp.pardir))
import sys

sys.path.insert(0, BASE_DIR)
from model.regressor.create_regressor import create_model
from utils.regressor.retail_dataset import RetailTest, parseList
from utils.regressor.distance_calculation_arcface import test_inference
from utils.regressor.retail_eval import evaluation_num_fold
from utils.config import IMAGE_RESOLUTION, CONCAT, PAIR_PATH, NUM_WORKERS, \
    TOTAL_PAIR, INTERVAL, MEAN
import torch
import numpy as np
import os.path as osp
from tqdm import tqdm


REGRESS_WEIGHT_PATH = osp.join(BASE_DIR, 'model_files/swin_small_cgd_epoch040._9.9633.ckpt')
BATCH_SIZE = 8


if __name__ == "__main__":
    regressor = create_model('swin_transformer', pretrained=False, input_size=IMAGE_RESOLUTION, cgd=True).cuda()
    regressor.load_state_dict(torch.load(REGRESS_WEIGHT_PATH)['net_state_dict'])
    regressor.eval()
    print('load regressor sucessfully!')

    img_size = IMAGE_RESOLUTION
    nl, nr, flags, folds = parseList(pair_path=PAIR_PATH)
    testdataset = RetailTest(nl, nr, img_size=img_size)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

    featureLs = []
    featureRs = []
    for data in tqdm(testloader):
        for i in range(len(data)):
            data[i] = data[i].cuda()
        features = [test_inference(d, regressor, concat=CONCAT, mean=MEAN).numpy() for d in data]
        featureLs.append(features[0])
        featureRs.append(features[1])
    featureLs = np.concatenate(featureLs, axis=0)
    featureRs = np.concatenate(featureRs, axis=0)

    result = {'fl': featureLs, 'fr': featureRs, 'fold': folds, 'flag': flags}
    # save tmp_result
    # scipy.io.savemat('./result/tmp_result.mat', result)
    accs, thresholds = evaluation_num_fold(result, num=TOTAL_PAIR / INTERVAL)
    accs = np.mean(accs)
    thresholds = np.mean(thresholds) 
    print('    ave: {:.4f}'.format(accs * 100))
    print('    best_threshold: {:.4f}'.format(thresholds))
    result = ('%10s' * 1 + '%10.4g' * 2) % (
        f'{osp.split(REGRESS_WEIGHT_PATH)}', accs, thresholds)
    with open('eval.txt', 'a') as f:
        f.write(result + '\n')
