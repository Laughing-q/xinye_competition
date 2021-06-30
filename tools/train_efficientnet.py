import os.path as osp
import sys
BASE_DIR = osp.abspath('.')
sys.path.insert(0, BASE_DIR)

from utils.regressor.config import BATCH_SIZE, SAVE_FREQ, RESUME, SAVE_DIR, \
            TEST_FREQ, TOTAL_EPOCH, MODEL_PRE, GPU, TRAIN_DIR, PAIR_PATH, TOTAL_PAIR, INTERVAL
from utils.regressor.retail_eval import evaluation_num_fold
from utils.regressor.retail_dataset import RetailTrain, RetailTest, parseList
from utils.regressor.distance_calculation_arcface import multi_image2embedding
from model.arcface import ArcMarginProduct

from torch import nn
from torch.nn import DataParallel
from datetime import datetime
from tqdm.autonotebook import tqdm
from torch.optim import lr_scheduler
import torch.optim as optim
import time
import numpy as np
import scipy.io
import timm
import os
import logging
import torch.utils.data

def init_log(output_dir):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        filename=os.path.join(output_dir, 'log.log'),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return logging

# gpu init
gpu_list = ''
multi_gpus = True
if isinstance(GPU, int):
    gpu_list = str(GPU)
else:
    multi_gpus = True
    for i, gpu_id in enumerate(GPU):
        gpu_list += str(gpu_id)
        if i != len(GPU) - 1:
            gpu_list += ','
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

# other init
start_epoch = 1
save_dir = os.path.join(SAVE_DIR, MODEL_PRE + 'v2_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
os.makedirs(save_dir, exist_ok=True)

logging = init_log(save_dir)
_print = logging.info

net = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=256)

# define trainloader and testloader
# img_size = net.get_image_size(model_name)
img_size = 112

trainset = RetailTrain(root=TRAIN_DIR, img_size=img_size)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=0, drop_last=True)
ArcMargin = ArcMarginProduct(in_features=256, out_features=trainset.class_nums)

# nl: left_image_path
# nr: right_image_path
nl, nr, flags, folds = parseList(pair_path=PAIR_PATH)
testdataset = RetailTest(nl, nr, img_size=img_size)
testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=0, drop_last=False)


if RESUME:
    ckpt = torch.load(RESUME)
    net.load_state_dict(ckpt['net_state_dict'])
    start_epoch = ckpt['epoch'] + 1


# define optimizers
optimizer_ft = optim.SGD(params=net.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=4e-4)

exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[36, 52, 58], gamma=0.1)


net = net.cuda()
ArcMargin = ArcMargin.cuda()
if multi_gpus:
    net = DataParallel(net)
    ArcMargin = DataParallel(ArcMargin)
criterion = torch.nn.CrossEntropyLoss()


best_acc = 0.0
best_epoch = 0
for epoch in range(start_epoch, TOTAL_EPOCH+1):
    exp_lr_scheduler.step()
    # train model
    _print('Train Epoch: {}/{} ...'.format(epoch, TOTAL_EPOCH))
    net.train()

    train_total_loss = 0.0
    total = 0
    since = time.time()
    trainloader = tqdm(trainloader)
    for i, data in enumerate(trainloader):
        img, label = data[0].cuda(), data[1].cuda()
        batch_size = img.size(0)
        optimizer_ft.zero_grad()

        raw_logits = net(img)

        output = ArcMargin(raw_logits, label)
        total_loss = criterion(output, label)
        total_loss.backward()
        optimizer_ft.step()

        train_total_loss += total_loss.item() * batch_size
        total += batch_size

        trainloader.set_description('Training Progress')
    train_total_loss = train_total_loss / total
    time_elapsed = time.time() - since
    loss_msg = '    total_loss: {:.4f} time: {:.0f}m {:.0f}s'\
        .format(train_total_loss, time_elapsed // 60, time_elapsed % 60)
    _print(loss_msg)

    # test model on lfw
    if epoch % TEST_FREQ == 0:
        net.eval()
        featureLs = []
        featureRs = []
        _print('Test Epoch: {} ...'.format(epoch))
        for data in tqdm(testloader):
            for i in range(len(data)):
                data[i] = data[i].cuda()
            features = [multi_image2embedding(d, net).numpy() for d in data]
            featureLs.append(features[0])
            featureRs.append(features[1])
        featureLs = np.concatenate(featureLs, axis=0)
        featureRs = np.concatenate(featureRs, axis=0)

        result = {'fl': featureLs, 'fr': featureRs, 'fold': folds, 'flag': flags}
        # save tmp_result
        # scipy.io.savemat('./result/tmp_result.mat', result)
        accs, thresholds = evaluation_num_fold(result, num=TOTAL_PAIR / INTERVAL)
        print(accs)
        _print('    ave: {:.4f}'.format(np.mean(accs) * 100))
        _print('    best_threshold: {:.4f}'.format(np.mean(thresholds) * 100))

    # save model
    if epoch % SAVE_FREQ == 0:
        msg = 'Saving checkpoint: {}'.format(epoch)
        _print(msg)
        if multi_gpus:
            net_state_dict = net.module.state_dict()
        else:
            net_state_dict = net.state_dict()
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save({
            'epoch': epoch,
            'net_state_dict': net_state_dict},
            os.path.join(save_dir, '%03d.ckpt' % epoch))
print('finishing training')
