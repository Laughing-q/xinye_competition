from pycocotools.coco import COCO
from PIL import Image
import os.path as osp
import numpy as np
import tqdm
import itertools
import random
import torch
import glob
import cv2
import os


def crop_images(pic_root, json_file, save_dir, suffix=''):
    os.makedirs(save_dir, exist_ok=True)
    anno = COCO(json_file)
    catIds = anno.getCatIds()
    for catId in tqdm.tqdm(catIds, total=len(catIds)):
        class_file = osp.join(save_dir, f"{int(catId)}")
        os.makedirs(class_file, exist_ok=True)
        imgId = anno.getImgIds(catIds=catId)  
        for ii, img_id in enumerate(imgId):
            img = anno.loadImgs(img_id)
            filename = img[0]['file_name']
            I = cv2.imread(os.path.join(pic_root, filename))
            ann_id = anno.getAnnIds(imgIds=img_id, catIds=catId, iscrowd=None)
            anns = anno.loadAnns(ann_id)
            for ai, ann in enumerate(anns):
                category_id = int(ann['category_id'])
                assert category_id == int(catId), 'The category_id is not same.'
                [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
                x1, y1, x2, y2 = bbox_x, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h
                cv2.imwrite(osp.join(class_file, f"{category_id}_{ii}_{ai}_{suffix}.jpg"), 
                            I[y1:y2, x1:x2, :])
    # count = 0
    # for img_id, annos in anno.imgToAnns.items():
    #     file_name = anno.imgs[img_id]['file_name']
    #     img_path = osp.join(pic_root, file_name)
    #     ori_img = cv2.imread(img_path)
    #     for ann in annos:
    #         category_id = int(ann['category_id'])
    #         x1, y1, w, h = [int(b) for b in ann['bbox']]
    #         cv2.imwrite(f"{category_id}_{count}", ori_img[y1:y1+h, x1:x1+w, :])
    #         count += 1

# def unique_random(a, b):

def gen_pair(pic_root, save_path='pair.txt', sim_ratio=0.5, total_num=10000, interval=500):
    """Generate pairs for test.

    Args:
        pic_root (str): The root of cropped test images, 
            pic_root/category_id/{category_id}_{img_id}_{id}_{suffix}.jpg .
    """
    cat_dirs = glob.glob(osp.join(pic_root, '*'))
    simIndex = random.choices(range(len(cat_dirs)), k=int(total_num * sim_ratio))
    # index2 = itertools.combinations(range(len(cat_dirs)), 2)
    count = 0
    with open(save_path, 'w') as f:
        # for i in simIndex:
        while count < int(sim_ratio * total_num):
            simIndex = random.randint(0, len(cat_dirs) - 1)
            img_names = glob.glob(osp.join(cat_dirs[simIndex], '*'))
            if len(img_names) < 2:
                continue
            pairs = sim_imgs(img_names)
            flag = 1
            f.write(f"{pairs[0]} {pairs[1]} {flag} {count // interval}" + '\n')
            count += 1
            print(count)
        while count < total_num:
            diffIndex1 = random.randint(0, len(cat_dirs) - 1)
            diffIndex2 = random.randint(0, len(cat_dirs) - 1)
            if diffIndex1 == diffIndex2:
                continue
            img_names1 = glob.glob(osp.join(cat_dirs[diffIndex1], '*'))
            img_names2 = glob.glob(osp.join(cat_dirs[diffIndex2], '*'))
            if len(img_names1) == 0 or len(img_names2) == 0: 
                continue
            pairs = diff_imgs(img_names1, img_names2)
            flag = -1
            f.write(f"{pairs[0]} {pairs[1]} {flag} {count // interval}" + '\n')
            count += 1
            print(count)


def sim_imgs(img_names):
    """
    Args:
        img_names (List): root/category_id/*
    """
    indexs = random.sample(range(len(img_names)), 2)
    return img_names[indexs[0]], img_names[indexs[1]]

def diff_imgs(img_names1, img_names2)   :
    """
    Args:
        img_names1 (List): root/category_id1/*
        img_names2 (List): root/category_id2/*
    """
    index1 = random.randint(0, len(img_names1) - 1)
    index2 = random.randint(0, len(img_names2) - 1)
    return img_names1[index1], img_names2[index2]

def parseList(pair_path):
    with open(pair_path, 'r') as f:
        pairs = [p.strip() for p in f.readlines()]

    random.shuffle(pairs)
    nameLs = []
    nameRs = []
    folds = []
    flags = []
    for i, p in enumerate(pairs):
        nameL, nameR, flag, fold = p.split(' ')
        nameLs.append(nameL)
        nameRs.append(nameR)
        flags.append(int(flag))
        folds.append(int(fold))

    return [nameLs, nameRs, np.array(flags), np.array(folds)]

# def parseList(pair_path):
#     with open(pair_path, 'r') as f:
#         pairs = [p.strip() for p in f.readlines()]
#     folder_name = 'cropped_test'
#     nameLs = []
#     nameRs = []
#     folds = []
#     flags = []
#     for i, p in enumerate(pairs):
#         p = p.split(' ')
#         if len(p) == 3:  # 同一个人的照片组成对比组
#             nameL = os.path.join(pair_path, folder_name, p[0], '{}.jpg'.format(int(p[1])))
#             nameR = os.path.join(pair_path, folder_name, p[0], '{}.jpg'.format(int(p[2])))
#             fold = i // 2400
#             flag = 1  # 正正对比
#         elif len(p) == 4:  # 不同人的照片组成对比组
#             nameL = os.path.join(pair_path, folder_name, p[0], '{}.jpg'.format(int(p[1])))
#             nameR = os.path.join(pair_path, folder_name, p[2], '{}.jpg'.format(int(p[3])))
#             fold = i // 2400
#             flag = -1  # 正负对比
#         nameLs.append(nameL)
#         nameRs.append(nameR)
#         folds.append(fold)
#         flags.append(flag)
#     # print(nameLs)
#     return [nameLs, nameRs, folds, flags]


class RetailDataset(object):
    """Generate database"""
    def __init__(self, pic_root, json_file):
        self.anno = COCO(json_file)
        # # check dataset
        # for k, v in self.anno.imgToAnns.items():
        #     if len(v) > 1:
        #         I = cv2.imread(os.path.join(
        #             pic_root, self.anno.imgs[k]['file_name']))
        #         I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
        #         plt.imshow(np.int16(I))
        #         self.anno.showAnns(v, draw_bbox=True)
        #         print(self.anno.imgs[k]['file_name'])
        #         plt.show()
        #         print('more than one retail.')
        # exit()
        self.image_list = []
        self.label_list = []
        self.bboxs = []
        for img_id, annos in self.anno.imgToAnns.items():
            file_name = self.anno.imgs[img_id]['file_name']
            img_path = osp.join(pic_root, file_name)
            for ann in annos:
                category_id = int(ann['category_id'])
                self.label_list.append(category_id)
                x1, y1, w, h = [int(b) for b in ann['bbox']]
                self.bboxs.append([x1, y1, w, h])
                self.image_list.append(img_path)

    def __getitem__(self, index):
        img_path = self.image_list[index]
        target = self.label_list[index]
        x1, y1, w, h = self.bboxs[index]
        img = np.array(Image.open(img_path))[y1:y1+h, x1:x1+w, :]
        img = cv2.resize(img, (112, 112))
        # cv2.imshow('p', img)
        # cv2.waitKey(0)

        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        # flip = np.random.choice(2)*2-1  # 1和-1随机爆
        # img = img[:, ::flip, :]  # 随机横向翻转
        # imglist = [img, img[:, ::-1, :]]
        # for i in range(len(imglist)):
        #     imglist[i] = (imglist[i] - 127.5) / 128.0
        #     imglist[i] = imglist[i].transpose(2, 0, 1)
        # imgs = [torch.from_numpy(i).float() for i in imglist]

        return img, target

    def __len__(self):
        return len(self.image_list)


class RetailTrain(object):
    """For training"""
    def __init__(self, root, img_size):
        self.root = root
        self.img_size = img_size

        # img_txt_dir = os.path.join(root, 'train.txt')
        # image_list = []
        # label_list = []
        # with open(img_txt_dir) as f:
        #     img_label_list = f.read().splitlines()
        # for info in img_label_list:
        #     image_dir, label_name = info.split(' ')
        #     image_list.append(os.path.join(root, 'train', image_dir))
        #     label_list.append(int(label_name))
        #
        # self.image_list = image_list
        # self.label_list = label_list
        # self.class_nums = len(np.unique(self.label_list))
        self.image_list = glob.glob(osp.join(self.root, '*/*'))
        self.label_list = [int(osp.basename(im).split('_')[0]) for im in self.image_list]
        self.class_nums = len(glob.glob(osp.join(self.root, '*')))

    def __getitem__(self, index):
        img_path = self.image_list[index]
        target = self.label_list[index]
        img = Image.open(img_path)
        img = img.resize((self.img_size, self.img_size))
        img = np.array(img)

        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        flip = np.random.choice(2)*2-1  # 1和-1随机
        img = img[:, ::flip, :]  # 随机横向翻转
        img = (img - 127.5) / 128.0
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        return img, target

    def __len__(self):
        return len(self.image_list)


class RetailTest(object):
    """For test"""
    def __init__(self, imgl, imgr, img_size):

        self.imgl_list = imgl
        self.imgr_list = imgr
        self.img_size = img_size

    def __getitem__(self, index):
        imgl = Image.open(self.imgl_list[index])
        imgl = imgl.resize((self.img_size, self.img_size))
        imgl = np.array(imgl)
        # imgl = scipy.misc.imread(self.imgl_list[index])
        if len(imgl.shape) == 2:  # 如果为灰度图
            imgl = np.stack([imgl] * 3, 2)  # 沿着2轴（通道处）连续stack三份，转为三通道

        imgr = Image.open(self.imgr_list[index])
        imgr = imgr.resize((self.img_size, self.img_size))
        imgr = np.array(imgr)
        # imgr = scipy.misc.imread(self.imgr_list[index])
        if len(imgr.shape) == 2:
            imgr = np.stack([imgr] * 3, 2)

        # imglist = [imgl, imgl[:, ::-1, :], imgr, imgr[:, ::-1, :]]  # 分别将l和r图片分别横向翻转，然后组成个四个元素的list
        imglist = [imgl, imgr]  # 分别将l和r图片分别横向翻转，然后组成个四个元素的list
        for i in range(len(imglist)):
            imglist[i] = (imglist[i] - 127.5) / 128.0
            imglist[i] = imglist[i].transpose(2, 0, 1)
        # imgs = [torch.from_numpy(i).float() for i in imglist]
        return imglist

    def __len__(self):
        return len(self.imgl_list)


if __name__ == "__main__":
    simiIndex = np.array(random.choices(range(1150), k=50000))
    simiIndex2 = np.array(random.choices(range(1150), k=50000))
    print((simiIndex == simiIndex2).sum())
    index2 = list(itertools.combinations(range(115), 2))
    print(len(index2))

    # index = list(itertools.combinations(range(5), 2))
    # print(index)
    exit()
    test_dataset = RetailDataset('/d/competition/retail/Preliminaries/test/b_images',
                                 '/d/competition/retail/Preliminaries/test/b_annotations.json')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=8, drop_last=False)

    for data in test_loader:
        print(data[0][0].shape)
        print(data[0][1].shape)
        print(data[1].shape)
