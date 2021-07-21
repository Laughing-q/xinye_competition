from .arcface import ArcMarginProduct
from .swin_transformer import SwinTransformer
from .cgd import CGDModel
import timm
from .CoAtNet import CoAtNet, DIMS, REPEAT_NUM
from .CircleLoss import SparseCircleLoss
from utils.config import FEATURE_DIMS



def create_model(name, pretrained, input_size, cgd=False):
    if name == 'efficientnet_b4':
        model = timm.create_model('efficientnet_b4', pretrained=pretrained, num_classes=FEATURE_DIMS)
    # for test training code
    elif name == 'mobilenetv3_large_100':
        model = timm.create_model('mobilenetv3_large_100', pretrained=pretrained, num_classes=FEATURE_DIMS)
    elif name == 'swin_transformer':
        model = SwinTransformer(img_size=input_size, num_classes=FEATURE_DIMS)
    elif name == 'CoATNet':
        model = CoAtNet(input_size, REPEAT_NUM['CoAtNet-0'], DIMS['CoAtNet-0'], class_num=FEATURE_DIMS)
    
    if cgd:
        return CGDModel(model, gd_config='SG', feature_dim=FEATURE_DIMS, num_classes=FEATURE_DIMS)
    else:
        return model


def create_metric(name):
    if name == 'Arcface':
        return ArcMarginProduct(in_features=256, out_features=FEATURE_DIMS)
    elif name == 'Circleloss':
        return SparseCircleLoss(m=0.25, emdsize=FEATURE_DIMS, class_num=FEATURE_DIMS, gamma=64, use_cuda=True)
