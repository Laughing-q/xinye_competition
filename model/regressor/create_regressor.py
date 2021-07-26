from .arcface import ArcMarginProduct
# from .swin_transformer import SwinTransformer
from .swin_transformer import swin_transformer
from .cgd import CGDModel
import timm
from .CoAtNet import CoAtNet, DIMS, REPEAT_NUM
from .CircleLoss import SparseCircleLoss
from utils.config import FEATURE_DIMS, SWIN_PRETRAIN, CLASS_NUM
import torch
import torch.nn as nn
from copy import deepcopy
import math


def create_model(name, pretrained, input_size, cgd=False):
    flag = 'conv'
    if name == 'efficientnet_b4':
        model = timm.create_model('efficientnet_b4', 
                                  pretrained=pretrained, num_classes=FEATURE_DIMS)
    # for test training code
    elif name == 'mobilenetv3_large_100':
        model = timm.create_model('mobilenetv3_large_100', 
                                  pretrained=pretrained, num_classes=FEATURE_DIMS)
    elif name == 'swin_transformer':
        model = swin_transformer(input_size=input_size, num_classes=FEATURE_DIMS, type='large')
        flag = 'swin'
        if pretrained:
            model.load_state_dict(torch.load(SWIN_PRETRAIN, map_location='cpu')['model'], strict=False)
    elif name == 'CoATNet':
        model = CoAtNet(input_size, REPEAT_NUM['CoAtNet-0'], 
                        DIMS['CoAtNet-0'], class_num=FEATURE_DIMS)
    if cgd:
        model = CGDModel(model, gd_config='SG', feature_dim=FEATURE_DIMS, 
                        num_classes=CLASS_NUM, flag=flag)
    return model


def create_metric(name):
    if name == 'Arcface':
        return ArcMarginProduct(in_features=FEATURE_DIMS, out_features=CLASS_NUM)
    elif name == 'Circleloss':
        return SparseCircleLoss(m=0.25, emdsize=FEATURE_DIMS, class_num=CLASS_NUM, gamma=64, use_cuda=True)

def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            # 对模型进行滑动平均
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        # 复制model的属性给EMA
        copy_attr(self.ema, model, include, exclude)
