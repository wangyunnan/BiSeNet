import cv2
import torch
import numpy as np
import os
import os.path as osp
import sys
import torch.nn.functional as F

from network.bisenet import BiSeNet
from utils.img_utils import normalize
from config import config

#Cityscapes Class Color Map (255 is change to 19)
color_map = {0: [128, 64, 128], 1: [244, 35, 232], 2: [70, 70, 70], 3: [102, 102, 156],
             4: [190, 153, 153], 5: [153, 153, 153], 6: [250, 170, 30], 7: [220, 220, 0],
             8: [107, 142, 35], 9: [152, 251, 152], 10: [70, 130, 180], 11: [220, 20, 60],
             12: [255, 0, 0], 13: [0, 0, 142], 14: [0, 0, 70], 15: [0, 60, 100], 16: [0, 80, 100],
             17: [0, 0, 230], 18: [119, 11, 32], 19: [0, 0, 0]}

class Transform(object):
    def __init__(self, img_mean, img_std):
        self.img_mean = img_mean
        self.img_std = img_std

    def __call__(self, img, gt):
        img = normalize(img, self.img_mean, self.img_std)
        img = img.transpose(2,0,1) #C * H * W

        return img, gt

class Visualizer(object):
    def __init__(self, ckpt_path='', save_path=''):
        #Init setting
        self.ckpt_path = ckpt_path
        self.save_path = save_path

        if not osp.exists(self.save_path):
            os.makedirs(self.save_path)

        self.save_names = []
        self.list_path = config.trainval_list
        self.root = config.dataset_root

        try:
            if not osp.exists(self.root): raise ValueError
        except ValueError :
            print('Data root is not exist!')
            sys.exit(0)

        self.num_classes = config.num_classes
        self.crop_size = config.eval_crop_size
        self.color_list = []
        self.stride_rate = config.eval_stride_rate

        for i in range(self.num_classes + 1):
            self.color_list.append(color_map[i])

        with open(self.list_path, 'r') as fr:
            self.files = fr.readlines()

        self.data_transform = Transform(config.img_mean, config.img_std)

        # Model setting
        self.model = BiSeNet(self.num_classes)
        self.model.load_state_dict(torch.load(self.ckpt_path, map_location='cuda:0'))
        print('Checkpoint loaded successfully!')
        self.model.cuda()
        self.model.eval()

    def visual_images(self, index):
        names = self.files[index[0]]
        index.remove(index[0])
        self.save_names = []
        imgs, gts = self.processing_files(names)

        for i in index:
            names = self.files[i]
            img, gt = self.processing_files(names)
            imgs = torch.cat((imgs,img), dim=0)
            gts = np.concatenate((gts, gt), axis=0)

        print(imgs.shape)
        prob = self.sliding_eval(imgs)
        pred = np.argmax(prob, axis=1)
        mask = (gts == 255)

        pred[mask] = self.num_classes

        color_list = np.asarray(self.color_list)
        show = color_list[pred.astype('int')]

        for i, img in enumerate(show):
            img = np.uint8(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow('visualization', img)
            cv2.waitKey(0)
            cv2.imwrite(osp.join(self.save_path, self.save_names[i]), img)

    def processing_files(self, names):
        img_name, gt_name = names.strip().split('\t')
        self.save_names.append(img_name.split('/')[-1].replace('_leftImg8bit.png','_bisenet_res18.png'))
        img_path = osp.join(self.root, img_name)
        gt_path = osp.join(self.root, gt_name)
        img = np.array(cv2.imread(img_path, cv2.IMREAD_COLOR))

        img = img[:, :, ::-1]  # bgr2rgb
        gt = np.array(cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE))

        if self.data_transform is not None:
            img, gt = self.data_transform(img, gt)

        img = torch.from_numpy(np.ascontiguousarray(img)).float()
        img = img.unsqueeze(0).cuda()
        gt = gt.astype(np.int64)[np.newaxis, :]

        return img, gt

    def sliding_eval(self, imgs):
        N, C, sH, sW = imgs.shape
        long_size = sH if sH > sW else sW

        if long_size <= self.crop_size:
            imgs, margin = self.shape_pad(imgs, (self.crop_size,) * 2)
            prob = self.inference(imgs)
            prob = prob[:, :, margin[0]:margin[1], margin[2]:margin[3]]
        else:
            stride = int(np.ceil(self.crop_size * self.stride_rate))
            imgs, margin = self.shape_pad(imgs, (self.crop_size,) * 2)

            pN, pC, pH, pW = imgs.shape
            nx = int(np.ceil((pW - self.crop_size) / stride)) + 1
            ny = int(np.ceil((pH - self.crop_size) / stride)) + 1

            prob = torch.zeros(pN, self.num_classes, pH, pW).cuda()

            for yi in range(ny):
                for xi in range(nx):
                    h_end = min(pH, stride * yi + self.crop_size)
                    w_end = min(pW, stride * xi + self.crop_size)
                    h_start = h_end - self.crop_size
                    w_start = w_end - self.crop_size

                    crop_imgs = imgs[:, :, h_start:h_end, w_start:w_end]
                    crop_prob = self.inference(crop_imgs)
                    prob[:, :, h_start:h_end, w_start:w_end] += crop_prob

            prob = prob[:, :, margin[0]:margin[1], margin[2]:margin[3]]
        prob = prob.detach().cpu().numpy()
        return prob

    def inference(self, img):
        out = self.model(img)[0]
        prob = F.softmax(out, 1)
        return prob

    def shape_pad(self, imgs, shape):
        N, C, H, W = imgs.shape
        margin = np.zeros(4, np.uint32)

        pad_h = shape[0] - H if shape[0] - H > 0 else 0
        pad_w = shape[1] - W if shape[1] - W > 0 else 0

        margin[0] = pad_h // 2
        margin[1] = pad_h // 2 + H
        margin[2] = pad_w // 2
        margin[3] = pad_w // 2 + W

        nH = margin[0] + margin[1] + pad_h % 2
        nW = margin[2] + margin[3] + pad_w % 2

        out = torch.zeros(N, C, nH, nW).cuda()
        out[:, :, margin[0]:margin[1], margin[2]:margin[3]] = imgs

        return out, margin

if __name__ == '__main__':
    # Visualization
    agent_visual = Visualizer(ckpt_path='./save/epoch-10.pth', save_path=osp.join(config.save_root, 'visual'))
    agent_visual.visual_images([2980,2])