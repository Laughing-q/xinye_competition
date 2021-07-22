import os.path as osp
import sys
BASE_DIR = osp.abspath(osp.join(osp.dirname(__file__), osp.pardir))
sys.path.insert(0, BASE_DIR)
from utils.config import TRAIN_SAVE_DIR, PAIR_PATH, \
    TOTAL_PAIR, INTERVAL, CONCAT, AUGMENT_PROBABILITY, NUM_WORKERS, save_args
from model.regressor.create_regressor import create_model, create_metric
from utils.regressor.retail_eval import evaluation_num_fold
from utils.regressor.retail_dataset import RetailTrain, RetailTest, parseList
from utils.regressor.plots import plot_recognition_results
from utils.regressor.distance_calculation_arcface import test_inference
from torch.nn import DataParallel
from datetime import datetime
from tqdm.autonotebook import tqdm
from torch.optim import lr_scheduler
import torch.optim as optim
import time
import numpy as np
import os
import os.path as osp
import logging
import torch.utils.data
import argparse
import yaml
from timm.scheduler.cosine_lr import CosineLRScheduler
import math

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=5, help='Total epochs')
parser.add_argument('--input_size', type=int, default=112, help='Input Size')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--resume', type=str, default='', help='Resume weights path')
parser.add_argument('--gpu', type=str, default='0, 1', help='GPUs')
parser.add_argument('--save_dir', type=str, default='./second_match', help='The path to save log and weights')
parser.add_argument('--name', type=str, default='Retail_v2_', help='save to save_dir/name')
parser.add_argument('--save_interval', type=int, default=1, help='The interval of saving model')
parser.add_argument('--test_interval', type=int, default=1, help='The interval of testing model')
parser.add_argument('--use_cgd', action='store_true', default=True, help='Whether to use CGD')
parser.add_argument('--backbone', default='efficientnet_b4', help='Backbone')
parser.add_argument('--loss_head', default='Circleloss', help='Loss_head')
parser.add_argument('--pretrain', default=False, help='start from pretrain model')
opt = parser.parse_args()


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
multi_gpus = False
if torch.cuda.device_count() > 1 and len(opt.gpu) > 1:
    multi_gpus = True
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

# other init
start_epoch = 1
save_dir = os.path.join(opt.save_dir, opt.backbone + '_' + opt.loss_head + '_' + str(opt.epochs) + '_' + str(opt.batch_size)
                        + '_' + str(opt.input_size) + '_' + str(opt.pretrain) + '_' + datetime.now().strftime('%Y%m%d_%H%M%S'))

os.makedirs(save_dir, exist_ok=True)

weights_dir = osp.join(save_dir, 'weights')
os.makedirs(weights_dir, exist_ok=True)

results_file = osp.join(save_dir, 'results.txt')

with open(osp.join(save_dir, 'opt.yaml'), 'w') as f:
    yaml.safe_dump(vars(opt), f, sort_keys=False)

with open(osp.join(save_dir, 'config.yaml'), 'w') as f:
    yaml.safe_dump(save_args, f, sort_keys=False)


logging = init_log(save_dir)
_print = logging.info
_print(opt)

img_size = opt.input_size

net = create_model(name=opt.backbone, pretrained=opt.pretrain, input_size=img_size, cgd=opt.use_cgd).cuda()

loss = create_metric(opt.loss_head).cuda()

# define trainloader and testloader
# img_size = net.get_image_size(model_name)

trainset = RetailTrain(root=TRAIN_SAVE_DIR, img_size=img_size, **AUGMENT_PROBABILITY)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size,
                                          shuffle=True, num_workers=NUM_WORKERS, drop_last=True)

# nl: left_image_path
# nr: right_image_path
nl, nr, flags, folds = parseList(pair_path=PAIR_PATH)
testdataset = RetailTest(nl, nr, img_size=img_size)
testloader = torch.utils.data.DataLoader(testdataset, batch_size=opt.batch_size,
                                         shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

if opt.resume:
    ckpt = torch.load(opt.resume)
    net.load_state_dict(ckpt['net_state_dict'])
    start_epoch = ckpt['epoch'] + 1

# define optimizers
optimizer_ft = optim.SGD(params=net.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=4e-4)
exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[5, 52, 90], gamma=0.1)
# optimizer_ft = optim.SGD(params=net.parameters(), lr=5e-4, momentum=0.9, nesterov=True, weight_decay=4e-4)
# optimizer_ft = optim.AdamW(params=net.parameters(), lr=5e-4, eps=1e-8,betas=(0.9, 0.999), weight_decay=0.05)
# exp_lr_scheduler = CosineLRScheduler(optimizer=optimizer_ft,
#                                      t_initial=opt.epochs * len(trainloader),
#                                      t_mul=1.,
#                                      lr_min=5e-6,
#                                      warmup_lr_init=5e-7,
#                                      warmup_t=math.ceil(0.05 * opt.epochs * len(trainloader)),
#                                      cycle_limit=1,
#                                      t_in_epochs=False
#                                      )

if multi_gpus:
    net = DataParallel(net)
    loss = DataParallel(loss)

best_acc = 0.0
best_epoch = 0
# result = ('%15s' * 4) % (
#         'epochs', 'train_loss', 'accs', 'thresholds')
#
# with open(results_file, 'a') as f:
#     f.write(result + '\n')  # append metrics, val_loss

# init metric and loss
train_total_loss, accs, thresholds = 0, 0, 0

for epoch in range(start_epoch, opt.epochs + 1):
    exp_lr_scheduler.step(epoch)
    # train model
    _print('Train Epoch: {}/{} ...'.format(epoch, opt.epochs))

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
        total_loss = loss(raw_logits, label)
        if multi_gpus:
            total_loss = torch.mean(total_loss)
        total_loss.backward()
        optimizer_ft.step()

        train_total_loss += total_loss.item() * batch_size
        total += batch_size

        trainloader.set_description('Training Progress')
    train_total_loss = train_total_loss / total
    time_elapsed = time.time() - since
    loss_msg = '    total_loss: {:.4f} time: {:.0f}m {:.0f}s' \
        .format(train_total_loss, time_elapsed // 60, time_elapsed % 60)
    _print(loss_msg)

    # test model on lfw
    if epoch % opt.test_interval == 0:
        net.eval()
        featureLs = []
        featureRs = []
        _print('Test Epoch: {} ...'.format(epoch))
        for data in tqdm(testloader):
            for i in range(len(data)):
                data[i] = data[i].cuda()
            features = [test_inference(d, net, concat=CONCAT).numpy() for d in data]
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
        _print('    ave: {:.4f}'.format(accs * 100))
        _print('    best_threshold: {:.4f}'.format(thresholds))

    result = ('%10s' * 1 + '%10.4g' * 3) % (
        f'{epoch}/{opt.epochs}', train_total_loss, accs, thresholds)

    with open(results_file, 'a') as f:
        f.write(result + '\n')  # append metrics, val_loss
    # save model
    if epoch % opt.save_interval == 0:
        msg = 'Saving checkpoint: {}'.format(epoch)
        _print(msg)
        if multi_gpus:
            net_state_dict = net.module.state_dict()
        else:
            net_state_dict = net.state_dict()
        torch.save({
            'epoch': epoch,
            'net_state_dict': net_state_dict},
            os.path.join(weights_dir, '%03d.ckpt' % epoch))

plot_recognition_results(save_dir=save_dir)
print('finishing training')
