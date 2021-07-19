from .distance_calculation_arcface import multi_image2embedding
from .retail_dataset import RetailDataset

from tqdm.autonotebook import tqdm
from PIL import Image

import matplotlib.pyplot as plt
import sys
import os
import os.path as osp
import numpy as np
import cv2
import scipy.io
import copy
import torch.utils.data
import argparse
import timm
import math

def getFeatureFromTorch(net, test_dataset, batch_size=1, concat=True):
    """获得特征向量"""
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False)

    Features = []
    ItermClass = []
    count = 0
    for imgs, targets in tqdm(test_loader):
        imgs = imgs.cuda()
        features = multi_image2embedding(imgs, net, concat=concat)
        Features.append(features)
        ItermClass.append(targets)

    Features = torch.cat(Features, dim=0).cpu()
    ItermClass = torch.cat(ItermClass, dim=0)
    result = {'feature': Features, 'class': ItermClass}
    return result
    # scipy.io.savemat(args.feature_save_dir, result)


def getAccuracy(scores, flags, threshold):
    """获得正正比大于阈值，正负比小于阈值的比例，即判断正确比例"""
    p = np.sum(scores[flags == 1] > threshold)  # flags用来区分正正对比还是正负对比
    n = np.sum(scores[flags == -1] < threshold)  # flags=1：正正，flags=-1：正负
    return 1.0 * (p + n) / len(scores)


def getThreshold(scores, flags, thrNum):
    """获得最佳阈值"""
    accuracys = np.zeros((2 * thrNum + 1, 1))  # [0, 0, 0 ... 0, 0, 0]
    thresholds = np.arange(-thrNum, thrNum + 1) * 1.0 / thrNum  # [-1.    -0.999 -0.998 ...  0.998  0.999  1.   ]
    for i in range(2 * thrNum + 1):
        accuracys[i] = getAccuracy(scores, flags, thresholds[i])  # 将threshold中模拟的值一个个扔进getAccuracy()中获得accuracy

    max_index = np.squeeze(accuracys == np.max(accuracys))  # 筛选出accuracy最大的索引
    bestThreshold = np.mean(thresholds[max_index])  # accuracy最大所对应的阈值取平均
    return bestThreshold


def evaluation_num_fold(result, num=20):
    """将测试集分成了10个组，分别对十个组进行evaluation"""
    num = math.ceil(num)
    ACCs = np.zeros(num)
    Thres = np.zeros(num)
    # result = scipy.io.loadmat(root)  # 加载.mat文件
    for i in tqdm(range(num)):  # n个组
        fold = result['fold']
        flags = result['flag']
        featureLs = result['fl']
        featureRs = result['fr']

        valFold = fold != i
        testFold = fold == i

        featureLs = featureLs / np.expand_dims(np.sqrt(np.sum(np.power(featureLs, 2), 1)), 1)  # 标准化
        featureRs = featureRs / np.expand_dims(np.sqrt(np.sum(np.power(featureRs, 2), 1)), 1)
        scores = np.sum(np.multiply(featureLs, featureRs), 1)

        # featureLs_normalized = f.normalize(torch.tensor(featureLs))
        # featureRs_normalized = f.normalize(torch.tensor(featureRs))
        # cosine = f.linear(featureLs_normalized, featureRs_normalized)
        # scores = np.array(cosine)

        threshold = getThreshold(scores[valFold], flags[valFold], 10000)
        ACCs[i] = getAccuracy(scores[testFold], flags[testFold], threshold)
        Thres[i] = threshold
    #     print('{}    {:.2f}'.format(i+1, ACCs[i] * 100))
    # print('--------')
    # print('AVE    {:.2f}'.format(np.mean(ACCs) * 100))
    return ACCs, Thres


if __name__ == '__main__':
    # getFeatureFromTorch(resume=args.resume)
    test_dataset = RetailDataset('/d/competition/retail/Preliminaries/test/b_images',
                                 '/d/competition/retail/Preliminaries/test/b_annotations.json')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=8, drop_last=False)

    for data in test_loader:
        print(len(data[0]))
        print(data[0][0].shape)
        print(data[1].shape)

