from .arcface import ArcMarginProduct
# from .swin_transformer import SwinTransformer
from .swin_transformer import swin_transformer
from .cgd import CGDModel
import timm
from .CoAtNet import CoAtNet, DIMS, REPEAT_NUM
from .CircleLoss import SparseCircleLoss
from utils.config import FEATURE_DIMS, SWIN_PRETRAIN, CLASS_NUM
import torch


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
        model = swin_transformer(input_size=input_size, num_classes=FEATURE_DIMS)
        flag = 'swin'
        if pretrained:
            model.load_state_dict(torch.load(SWIN_PRETRAIN, map_location='cpu')['model'], strict=False)
    elif name == 'CoATNet':
        model = CoAtNet(input_size, REPEAT_NUM['CoAtNet-0'], 
                        DIMS['CoAtNet-0'], class_num=FEATURE_DIMS)
    if cgd:
        return CGDModel(model, gd_config='SG', feature_dim=FEATURE_DIMS, 
                        num_classes=CLASS_NUM, flag=flag)
    else:
        return model


def create_metric(name):
    if name == 'Arcface':
        return ArcMarginProduct(in_features=FEATURE_DIMS, out_features=CLASS_NUM)
    elif name == 'Circleloss':
        return SparseCircleLoss(m=0.25, emdsize=FEATURE_DIMS, class_num=CLASS_NUM, gamma=64, use_cuda=True)
