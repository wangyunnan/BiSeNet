import torch
from torch.utils.data import DataLoader

from config import config
from datasets.cityscapes import Cityscapes
from utils.img_utils import color_jitter, random_mirror, random_scale, normalize,random_crop

class TrainTransform(object):
    def __init__(self, img_mean, img_std, img_scale, crop_size):
        self.img_mean = img_mean
        self.img_std = img_std
        self.img_scale = img_scale
        self.crop_size = crop_size

    def __call__(self, img, gt):
        img = color_jitter(img)
        img, gt = random_mirror(img, gt)
        img, gt = random_scale(img, gt, self.img_scale)
        img = normalize(img, self.img_mean, self.img_std)
        img, gt = random_crop(img, gt, self.crop_size)
        img = img.transpose(2,0,1) #C * H * W

        return img, gt

def get_train_loader(dataset):
    data_setting = {'root': config.dataset_root,
                    'train_list': config.train_list,
                    'val_list': config.val_list}

    data_transform = TrainTransform(config.img_mean, config.img_std,
                                    config.train_scale, config.crop_size)

    train_dataset = dataset(data_setting, "train", data_transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              drop_last=True,
                              shuffle=False,
                              pin_memory=True,
                              sampler=train_sampler
                              )

    return train_loader, train_sampler

class EvalTransform(object):
    def __init__(self, img_mean, img_std):
        self.img_mean = img_mean
        self.img_std = img_std

    def __call__(self, img, gt):
        img = normalize(img, self.img_mean, self.img_std)
        img = img.transpose(2,0,1) #C * H * W

        return img, gt

def get_eval_loader(dataset):
    data_setting = {'root': config.dataset_root,
                    'train_list': config.train_list,
                    'val_list': config.val_list}

    data_transform = EvalTransform(config.img_mean, config.img_std)

    eval_dataset = dataset(data_setting, "val", data_transform)
    eval_loader = DataLoader(eval_dataset,
                              batch_size=2,
                              num_workers=config.num_workers,
                              drop_last=False,
                              shuffle=False,
                              )

    return eval_loader

if __name__ == '__main__':
    train_loader = get_eval_loader(Cityscapes)