from easydict import EasyDict
import os.path as osp
import torch
import time
import math

C = EasyDict()
config = C

"""Save Seting"""
time_cur = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
C.save_root = osp.join('./save', time_cur)
C.snapshot_epoch = 50
C.log_iter = 50

"""Dataset Setting"""
C.dataset_root = '/home/wangyunnan/datasets/Cityscapes'
C.train_list = './datasets/cityscapes/list/train.txt'
C.val_list = './datasets/cityscapes/list/val.txt'
C.test_list = './datasets/cityscapes/list/test.txt'
C.trainval_list = './datasets/cityscapes/list/trainval.txt'

"""Image Setting"""
C.num_classes = 19
C.ignore_label = 255
C.img_mean = [0.485, 0.456, 0.406]
C.img_std = [0.229, 0.224, 0.225]
C.crop_size = [1024, 1024]
C.seed = 12345

"""Train Setting"""
C.lr = 1e-2
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 5e-4
C.batch_size = 8
C.num_gpus = torch.cuda.device_count()
C.num_iters = 80000
C.num_epochs = math.ceil(C.num_iters / math.ceil(2975 / (C.batch_size * C.num_gpus)))
C.num_workers = 6
C.train_scale = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

"""Eval Setting"""
C.eval_stride_rate = 5 / 6.
C.eval_scale = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]  # multi scales: 0.5, 0.75, 1, 1.25, 1.5, 1.75
C.eval_flip = True  # True if use the ms_flip strategy
C.eval_crop_size = 1024

if __name__ == '__main__':
    print(C.num_epochs)

