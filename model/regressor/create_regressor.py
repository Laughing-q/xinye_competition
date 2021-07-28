from .arcface import ArcMarginProduct
# from .swin_transformer import SwinTransformer
from .swin_transformer import swin_transformer
from .cgd import CGDModel
import timm
from .CoAtNet import CoAtNet, DIMS, REPEAT_NUM
from .CircleLoss import SparseCircleLoss
from utils.config import FEATURE_DIMS, SWIN_PRETRAIN, CLASS_NUM,\
    IMAGE_RESOLUTION, CONCAT, MEAN
import torch
import torch.nn as nn
from copy import deepcopy
import math
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F


def create_model(name, pretrained, input_size, cgd=False, swin_type='large'):
    flag = 'conv'
    if name == 'efficientnet_b4':
        model = timm.create_model('efficientnet_b4', 
                                  pretrained=pretrained, num_classes=FEATURE_DIMS)
    # for test training code
    elif name == 'mobilenetv3_large_100':
        model = timm.create_model('mobilenetv3_large_100', 
                                  pretrained=pretrained, num_classes=FEATURE_DIMS)
    elif name == 'swin_transformer':
        model = swin_transformer(input_size=input_size, num_classes=FEATURE_DIMS, type=swin_type)
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


class Regressor(nn.Module):
    def __init__(self, model_name, pretrained=False, cgd=False, swin_type='large'):
        super(Regressor, self).__init__()
        self.model = create_model(model_name, pretrained, input_size=IMAGE_RESOLUTION,
                             cgd=cgd, swin_type=swin_type).cuda()
        self.concat = CONCAT
        self.mean = MEAN
        self.train = False
        if self.concat and self.mean:
            raise ValueError(f"`concat` and `mean` are mutually exclusive.")

    def train(self):
        self.train = True
        self.concat = False
        self.mean = False
        self.model.train()

    def eval(self):
        self.train = False
        self.concat = CONCAT
        self.mean = MEAN
        self.model.eval()

    def load_weight(self, weight_path):
        """load weight"""
        self.model.load_state_dict(torch.load(weight_path)['net_state_dict'])
        self.eval()

    def toTensor(self, imgs, device='cuda'):
        """preprocess, normalization and permutation"""
        if isinstance(imgs, list):
            imglist = np.stack(imgs, axis=0) if len(imgs) > 1 else imgs[0][None, ...]
            imgs = torch.from_numpy(imglist)
        elif isinstance(imgs, np.ndarray):
            imgs = torch.from_numpy(imgs)
        elif isinstance(imgs, torch.Tensor):
            imgs = imgs
        else:
            raise TypeError("The type of imgs should be list, np.ndarray or torch.Tensor,"
                            f"but got {type(imgs)}")
        if imgs.ndim == 3:
            imgs = imgs[None, ...]

        imgs = imgs.cuda().float()
        imgs = (imgs - 127.5) / 128.
        imgs = imgs.permute(0, 3, 1, 2).contiguous()  # bchw
        return imgs


    def generateBase(self, test_loader):
        """Generate the database features
        Args:
            test_loader (torch.utils.data.DataLoader): The test dataloader.

        Returns:
            result (Dict): {'features': , 'class': }, the features and class.
        """
        Features = []
        ItermClass = []
        for imgs, targets in tqdm(test_loader):
            imgs = self.toTensor(imgs)
            features = self.batched_inference(imgs)
            Features.append(features)
            ItermClass.append(targets)

        Features = torch.cat(Features, dim=0).cpu()
        ItermClass = torch.cat(ItermClass, dim=0)
        result = {'feature': Features, 'class': ItermClass}
        return result

    def flip_inference(self, imgs):
        """Inference with flip image, then concat or mean.
        Args:
            imgs (torch.Tensor): images after normalization and permute, range[-1, 1], NCHW

        Returns:
            return the output features.
        """
        imgs = torch.cat([imgs, imgs.flip(dims=[-1])], dim=0)  # (2N, C, H, W)
        res = self.model(imgs)  # (2N, 128)
        sub_dim = res.shape[-1]
        res = torch.split(res, res.shape[0] // 2, dim=0)  # ((N, 128), (N, 128))
        res = torch.stack(res, dim=0)  # (2, N, 128)
        res = res.permute(1, 0, 2).contiguous()  # (N, 2, 128)
        if self.concat:
            res = res.view(-1, sub_dim * 2)  # (N, 256)
        else:
            res = torch.mean(res, dim=-2).view(-1, sub_dim)
        features = res.data.cpu()  # .numpy()
        return features


    def get_features(self, imgs):
        """Two style of inference

        Args:
            imgs (torch.Tensor): images after normalization and permute, range[-1, 1], NCHW

        Returns:
            return the output features.
        """
        if self.concat or self.mean:
            features = self.flip_inference(imgs)
            # imgs = torch.cat([imgs, 
            #                   imgs.flip(dims=[-1]), 
            #                   imgs.flip(dims=[-2]),
            #                   imgs.flip(dims=[-1, -2])], dim=0)  # (2N, C, H, W)
            # res = self.model(imgs)  # (2N, 128)
            # sub_dim = res.shape[-1]
            # res = torch.split(res, res.shape[0] // 4, dim=0)  # ((N, 128), (N, 128))
            # res = torch.stack(res, dim=0)  # (2, N, 128)
            # res = res.permute(1, 0, 2).contiguous()  # (N, 2, 128)
            # res = res.view(-1, sub_dim * 4)  # (N, 256)
            # features = res.data.cpu()  # .numpy()
        else:
            res = self.model(imgs)  # (N, features_dim)
            features = res.data.cpu()  # .numpy()
        return features


    def batched_inference(self, imgs, batch_size=None):
        """Inference for a end-to-end pipeline with batch size.

        Args:
            imgs (torch.Tensor): images after normalization and permute, range[-1, 1], NCHW
            batch_size (int): batch size <= N.

        Returns:
            return the output features, (N, features_dim)
        """
        if batch_size is not None:
            features = []
            for sub_imgs in torch.split(imgs, batch_size, dim=0):
                sub_features = self.get_features(sub_imgs)
                features.append(sub_features)
            features = torch.cat(features, dim=0)
        else:
            features = self.get_features(imgs)

        return features

    def forward(self, imgs, batch_size=None):
        """A implement, to keep the same with Ensemble
        Args:
            imgs (List[np.array] | np.ndarray | torch.Tensor): images after resized, 
                before the normalization and permutation, range[0, 255], NHWC.
            batch_size (int): batch size.
            ensemble (bool): If True, no preprocessing. Casue the preprocess has been done
                for multiple model preprocessing once.

        Returns:
            return the output features.
        """
        if self.train:
            imgs = imgs
        else:
            imgs = self.toTensor(imgs)
        return self.batched_inference(imgs, batch_size)

    def test_inference(self, imgs):
        """ Inference for eval.

        Args:
            img (List[np.array] | np.ndarray | torch.Tensor): Resized, normalized input images.
        """
        # imgs = imgflip(imgs)  # (2N, C, H, W)
        # bhwc
        if isinstance(imgs, list):
            imglist = np.stack(imgs, axis=0) if len(imgs) > 1 else imgs[0][None, ...]
            imgs = torch.from_numpy(imglist)
        elif isinstance(imgs, np.ndarray):
            imgs = torch.from_numpy(imgs)
        elif isinstance(imgs, torch.Tensor):
            imgs = imgs
        else:
            raise TypeError("The type of imgs should be list, np.ndarray or torch.Tensor,"
                            f"but got {type(imgs)}")
        if imgs.ndim == 3:
            imgs = imgs[None, ...]

        imgs = imgs.cuda().float()
        features = self.get_features(imgs)

        return features

    def matching(self, database, category, features):
        """Matching the category through database and features
        Args:
            database (torch.Tensor): the database features generated by `func: generateBase`.
            category (torch.Tensor): the database category generated by `func: generateBase`.
            features (torch.Tensor): the pred features generated by model.
        """
        features_normalized = F.normalize(features)
        database_normalized = F.normalize(database)
        cosine = F.linear(features_normalized, database_normalized)
        similarity = cosine.permute(1, 0).contiguous()  # .numpy()

        best_similarity, index = torch.max(similarity, dim=0)
        result_category = category[index]

        return result_category, best_similarity     # num_imgs

    def selectResutlt(self, categories, scores, method='max'):
        """A implement, to keep the same with Ensemble, select pred result."""
        return categories.tolist(), scores.tolist()


class Ensemble(nn.ModuleList):
    """Inference Only"""
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, imgs, batch_size=None, fuse_feature=False):
        """
        Args:
            imgs (List[np.array] | np.ndarray | torch.Tensor): images after resized, 
                before the normalization and permutation, range[0, 255], NHWC.
            batch_size (int): batch size.

        Returns:
            return the output features.
        """
        imgs = self[0].toTensor(imgs)
        features = []
        for module in self:
            features.append(module.batched_inference(imgs, batch_size=batch_size))
        total_features = torch.stack(features, dim=0)  # [num_models, num_images, features_dim]
        if fuse_feature:
            return torch.mean(total_features, dim=0) # [num_images, features_dim]
        else:
            return total_features

    def generateBase(self, test_loader):
        """Generate the database features
        Args:
            test_loader (torch.utils.data.DataLoader): The test dataloader.

        Returns:
            result (Dict): {'features': , 'class': }, the features and class.
        """
        Features = []
        ItermClass = []
        for imgs, targets in tqdm(test_loader):
            # features = self.batched_inference(imgs)
            features = self.forward(imgs)
            Features.append(features)
            ItermClass.append(targets)

        Features = torch.cat(Features, dim=1).cpu()  # [num_models, num_imgs, features_dim]
        ItermClass = torch.cat(ItermClass, dim=0)
        result = {'feature': Features, 'class': ItermClass}
        return result

    def matching(self, database, category, features):
        """Matching the category through database and features
        Args:
            database (torch.Tensor): the database features generated by `func: generateBase`.
            category (torch.Tensor): the database category generated by `func: generateBase`.
            features (torch.Tensor): the pred features generated by model.
        """
        pred_categories, best_similarities = [], []
        for i, (db, feature) in enumerate(zip(database, features)):
            result_category, best_similarity = self[i].matching(db, category, feature)
            pred_categories.append(result_category)
            best_similarities.append(best_similarity)
        pred_categories = torch.stack(pred_categories, dim=0)
        best_similarities = torch.stack(best_similarities, dim=0)
        return pred_categories, best_similarities     # num_models, num_imgs


    def selectResutlt(self, categories, scores, method='vote'):
        """Select the best result from multiple model predictions
        Args:
            categories (torch.Tensor): categories from multiple models.
            scores (torch.Tensor): scores from multiple models.
            method (str): The way to select the best result.
        Returns:
            the selected results (tuple(List[], List[])).
        """
        if method not in ['max', 'vote']:
            raise ValueError(f"expected `max` or `vote`, but got {method}")
        if method == 'max' or len(self) == 2:
            # select the category with maximum similarity
            _, index = torch.max(scores, dim=0)
            rows, cols = index, torch.tensor(range(len(index)))
            final_categories = categories[rows, cols].tolist()
            final_scores = scores[rows, cols].tolist()
        else:
            # select the category with more votes.
            # supported odd num_models for now, cause even num_models
            # may get same bincount.
            final_categories = []
            final_scores = []
            for score, categiry in zip(scores.T, categories.T):
                bincount = torch.bincount(categiry)
                _, index = torch.max(bincount, dim=0)
                best_score = score[categiry==index].max()
                final_categories.append(index)
                final_scores.append(best_score)

        return final_categories, final_scores


def load_regressor(weights, 
                   model_names, 
                   pretrained,
                   cgd,
                   swin_type):
    models = Ensemble()
    weights = weights if isinstance(weights, list) else [weights]
    model_names = model_names if isinstance(model_names, list) else [model_names]
    pretrained = pretrained if isinstance(pretrained, list) else [pretrained]
    cgd = cgd if isinstance(cgd, list) else [cgd]
    swin_type = swin_type if isinstance(swin_type, list) else [swin_type]

    for w, m, p, c, s in zip(weights, model_names, pretrained, cgd, swin_type):
        regressor = Regressor(m, p, c, s)
        regressor.load_weight(w)
        models.append(regressor)
    
    if len(models) == 1:
        return models[-1]  # return model
    return models



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
