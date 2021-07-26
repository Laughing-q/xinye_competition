from utils.torch_utils import select_device, load_classifier, time_synchronized, time_synchronized
from model.yolo import Model
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.ensemble_boxes_wbf import weighted_boxes_fusion
from utils.plots import plot_one_box
from utils.general import non_max_suppression, scale_coords

import random
import os
import os.path as osp
import cv2
import numpy as np
import torch
import glob
import yaml


class Yolov5:
    def __init__(self, cfg, weight_path, device, img_hw=(384, 640)):
        self.weights = weight_path
        self.device = select_device(device)
        self.half = True
        # # path aware
        # self.model = attempt_load(self.weights, map_location=self.device)
        # self.model = torch.load(self.weights, map_location=self.device)['model'].float().fuse().eval()

        # # pt -> pth, path agnostic
        # self.model = torch.load(self.weights, map_location=self.device)['model']
        # with open(weight_path.replace('.pt', '.yaml'), 'w') as f:
        #     yaml.safe_dump(self.model.yaml, f, sort_keys=False)
        # torch.save(self.model.float().state_dict(), weight_path.replace('.pt', '.pth'))
        # self.model.float().fuse().eval()

        # # path agnostic
        ckpt = torch.load(self.weights, map_location=None)  # load checkpointA
        self.model = Model(cfg).to(self.device)
        with torch.no_grad():
            self.model.load_state_dict(ckpt, strict=False)  # load
            self.model.float().fuse().eval()

        if self.half:
            self.model.half()  # to FP16
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)]
                       for _ in range(len(self.names))]
        self.show = False
        self.img_hw = img_hw
        self.pause = False

    def preprocess(self, image, auto=True):  # (h, w)
        if type(image) == str and os.path.isfile(image):
            img0 = cv2.imread(image)
        else:
            img0 = image
        # img, _, _ = letterbox(img0, new_shape=new_shape)
        img, _, _ = letterbox(img0, new_shape=self.img_hw, auto=auto)
        # cv2.imshow('x', img)
        # cv2.waitKey(0)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        return img, img0

    def dynamic_detect(self, image, img0s, classes=None, conf_threshold=0.6, iou_threshold=0.4, agnostic=False, wbf=False):
        """
        Detect images by yolov5.

        Args:
            image (np.ndarray): a batch of images after self.preprocess(resize and pad), (b, c, h, w).
            image_raw (list[np.ndarray]): a batch of original images(original height and width).
            classes (list[int] | None): index of classes to fliter the classes.
            conf_threshold (float): confidence threshold.
            iou_threshold (float): iou threshold.
            agnostic (bool): Whether to ingore class when doing nms.

        Returns:
            pred (list[torch.Tensor]): predicted results, each tensor (..., 6) if predicted something
                else None.
        """
        img = torch.from_numpy(image).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # print(img.shape)
        torch.cuda.synchronize()
        pred = self.model(img)[0] 
        pred = non_max_suppression(pred, conf_threshold, iou_threshold, 
                                   classes=classes, agnostic=agnostic, 
                                   wbf=wbf, img_size=img.shape[2:])

        torch.cuda.synchronize()
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], img0s[i].shape).round()
                for di, (*xyxy, conf, cls) in enumerate(reversed(det)):
                    # label = '%s %.2f' % (self.names[int(cls)], conf)
                    # label = '%s' % (self.names[int(cls)])
                    label = '%.2f' % (conf)
                    # xyxy = [int(i) for i in xyxy]
                    # im0[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2], :] = 114
                    # if not self.names[int(cls)] in ['uniform', 'no-uniform']:
                    if self.show:
                        plot_one_box(xyxy, img0s[i], label=label,
                                     color=self.colors[int(cls)], 
                                     line_thickness=2)

        if self.show:
            for i in range(len(img0s)):
                cv2.namedWindow(f'p{i}', cv2.WINDOW_NORMAL)
                cv2.imshow(f'p{i}', img0s[i])
            key = cv2.waitKey(0 if self.pause else 1)
            self.pause = True if key == ord(' ') else False
            if key == ord('q') or key == ord('e') or key == 27:
                exit()
        return pred


if __name__ == "__main__":
    detector = Yolov5(weight_path='/d/projects/yolov5_arcface/weights/yolov5s_best.pt', device='0', img_hw=(640, 640))
    detector.show = True

    img_paths = glob.glob('/d/competition/retail/Preliminaries/test/a_images/*')
    for path in img_paths:
        img = cv2.imread(path)
        file_name = osp.basename(path)

        img, img_raw = detector.preprocess(img, auto=True)
        preds = detector.dynamic_detect(img, [img_raw])


