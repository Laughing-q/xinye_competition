import torch
import time
import glob
import scipy.io
import cv2
import numpy as np
from PIL import Image
import timm
import torch.nn.functional as F


def imgflip(imgs):
    """
    Args:
        img (List[np.array]): Resized(96, 112) input images.
    """
    # bhwc
    imglist = np.stack(imgs, axis=0) if len(imgs) > 1 else imgs[0][None, ...]
    imgs = torch.from_numpy(imglist).cuda().float()
    imgs = (imgs - 127.5) / 128.
    imgs = imgs.permute(0, 3, 1, 2).contiguous()  # bchw
    imgs = torch.cat([imgs, imgs.flip(dims=[-1])], dim=0)

    return imgs

def multi_image2embedding(imgs, net, batch_size=None):
    """Inference for a end-to-end pipeline.

    Args:
        img (List[np.array] | np.ndarray | torch.Tensor): Resized input images,
            without normalization and permute.
        net (Model): The regression model.
        batch_size (int): batch size.
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

    if batch_size is not None:   
        features = []
        for sub_imgs in torch.split(imgs, batch_size, dim=0):
            sub_imgs = sub_imgs.cuda().float()
            sub_imgs = (sub_imgs - 127.5) / 128.
            sub_imgs = sub_imgs.permute(0, 3, 1, 2).contiguous()  # bchw
            sub_imgs = torch.cat([sub_imgs, sub_imgs.flip(dims=[-1])], dim=0)  # (2N, C, H, W)

            res = net(sub_imgs)  # (2N, 128)
            res = torch.split(res, res.shape[0] // 2, dim=0) # ((N, 128), (N, 128))
            res = torch.stack(res, dim=0)  # (2, N, 128)
            res = res.permute(1, 0, 2).contiguous()  # (N, 2, 128)
            res = res.view(-1, 512)  # (N, 256)
            sub_features = res.data.cpu()#.numpy()
            features.append(sub_features)

        features = torch.cat(features, dim=0)
    else:
        imgs = imgs.cuda().float()
        imgs = (imgs - 127.5) / 128.
        imgs = imgs.permute(0, 3, 1, 2).contiguous()  # bchw
        imgs = torch.cat([imgs, imgs.flip(dims=[-1])], dim=0)  # (2N, C, H, W)

        res = net(imgs)  # (2N, 128)
        res = torch.split(res, res.shape[0] // 2, dim=0) # ((N, 128), (N, 128))
        res = torch.stack(res, dim=0)  # (2, N, 128)
        res = res.permute(1, 0, 2).contiguous()  # (N, 2, 128)
        res = res.view(-1, 512)  # (N, 256)
        features = res.data.cpu()#.numpy()

    return features

def multi_matching(img, database, category, net, batch_size=20):
    """Match the inference results.

    Args:
        img (List[np.array]): Resized input images.
        database (torch.Tensor): The databese.
        category (torch.Tensor): The category.
        net (Model): The regression model.
        batch_size (int): batch size.
    """
    features= multi_image2embedding(img, net, batch_size)  # (N, 256)

    # find the similarity
    features_normalized = F.normalize(features)
    database_normalized = F.normalize(torch.from_numpy(database))
    cosine = F.linear(features_normalized, database_normalized)
    similarity = cosine.permute(1, 0).contiguous().numpy()

    # features = features / np.expand_dims(np.sqrt(np.sum(np.power(features, 2), 1)), 1)  # (N, 256)
    # database = database / np.expand_dims(np.sqrt(np.sum(np.power(database, 2), 1)), 1)  # (M, 256)
    # scores = database@features.T   # (M, N)

    best_similarity = np.max(similarity, axis=0)  # (N, )
    index = np.argmax(similarity, axis=0)  # (N, )
    result_categories = category[index]

    return result_categories, best_similarity.round(5)


def test_inference(imgs, net):
    """ Inference for eval.

    Args:
        img (List[np.array] | np.ndarray | torch.Tensor): Resized, normalized input images.
        net (Model): The regression model.
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
    # imgs = (imgs - 127.5) / 128.
    # imgs = imgs.permute(0, 3, 1, 2).contiguous()  # bchw
    imgs = torch.cat([imgs, imgs.flip(dims=[-1])], dim=0)  # (2N, C, H, W)

    res = net(imgs)  # (2N, 128)
    res = torch.split(res, res.shape[0] // 2, dim=0) # ((N, 128), (N, 128))
    res = torch.stack(res, dim=0)  # (2, N, 128)
    res = res.permute(1, 0, 2).contiguous()  # (N, 2, 128)
    res = res.view(-1, 512)  # (N, 256)
    features = res.data.cpu()#.numpy()

    return features

if __name__ == '__main__':
    img_paths = glob.glob('/d/competition/retail/Preliminaries/train/search_images/b/86,weiweidounai/*')
    # regressor = model.MobileFacenet()
    # regressor.load_state_dict(torch.load('/d/projects/yolov5_arcface/weights/arcface_best.ckpt')['net_state_dict'])
    # regressor.cuda()
    # regressor.eval()
    regressor = timm.create_model('efficientnet_b4', pretrained=False, num_classes=256).cuda()
    regressor.load_state_dict(torch.load('./weights/efficient_99.90.ckpt')['net_state_dict'])
    regressor.eval()

    categories = []
    scores = []

    # mat = scipy.io.loadmat('matrix_features/resnet_112.mat')   # (M, 256)
    mat = scipy.io.loadmat('matrix_features/efficient_99.90.mat')   # (M, 256)
    database = mat['feature']

    category = mat['class'][0]

    ts = time.time()
    imgs = []
    for img_path in img_paths:
        print(img_path.split('/')[-2])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112, 112))

        # category, score = matching(img,
        #                            mat_path='matrix_features/efficient_99.90.mat',
        #                            net=regressor)
        # categories.append(category)
        # scores.append(score)

        imgs.append(img)
    categories, scores = multi_matching(imgs,
                               database=database,
                               category=category,
                               net=regressor)
    print(np.array(categories))
    print(np.array(scores))
    print('total time:', time.time() - ts)
