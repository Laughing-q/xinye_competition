from .arcface import ArcMarginProduct
from .swin_transformer import SwinTransformer
import timm
from .CoAtNet import CoAtNet
from .CircleLoss import SparseCircleLoss
from utils.config import IMAGE_RESOLUTION, FEATURE_DIMS, DIMS, REPEAT_NUM



def create_model(name, pretrained):
    if name == 'efficientnet_b4':
        return timm.create_model('efficientnet_b4', pretrained=pretrained, num_classes=FEATURE_DIMS)
    elif name == 'swin_transformer':
        return SwinTransformer(img_size=IMAGE_RESOLUTION, num_classes=FEATURE_DIMS)
    elif name == 'CoATNet':
        return CoAtNet(IMAGE_RESOLUTION, REPEAT_NUM['CoAtNet-0'], DIMS['CoAtNet-0'], class_num=FEATURE_DIMS)

def create_metric(name):
    if name == 'Arcface':
        return ArcMarginProduct(in_features=256, out_features=FEATURE_DIMS)
    elif name == 'Circleloss':
        return SparseCircleLoss(m=0.25, emdsize=FEATURE_DIMS, class_num=FEATURE_DIMS, gamma=64, use_cuda=True)
