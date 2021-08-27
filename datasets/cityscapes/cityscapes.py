import torch
from torch.utils.data import Dataset
import os.path as osp
import numpy as np
import cv2

class Cityscapes(Dataset):
    def __init__(self, data_setting, mode, data_transform=None):
        super(Cityscapes, self).__init__()
        self.root = data_setting['root']
        self.train_list_path = data_setting['train_list']
        self.val_list_path = data_setting['val_list']
        self.mode = mode
        self.file_names = self.get_file_names(self.mode)
        self.data_transform = data_transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        names = self.file_names[index]
        img_path = osp.join(self.root, names[0])
        gt_path = osp.join(self.root, names[1])

        img = np.array(cv2.imread(img_path, cv2.IMREAD_COLOR))
        img = img[:, :, ::-1]  # bgr2rgb
        gt =  np.array(cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE))
        if self.data_transform is not None:
            img, gt = self.data_transform(img, gt)

        img = torch.from_numpy(np.ascontiguousarray(img)).float()
        gt = torch.from_numpy(np.ascontiguousarray(gt)).long()

        return img, gt

    def get_file_names(self, mode):
        assert mode in ['train', 'val']
        list_path = self.train_list_path if mode == 'train' else self.val_list_path
        file_names = []
        with open(list_path, 'r') as fr:
            files = fr.readlines()
        for item in files:
            img_name, gt_name = item.strip().split('\t')
            file_names.append([img_name, gt_name])

        return file_names

if __name__ == '__main__':
    print('In cityscapes')

    data_setting = {'root': '',
                    'train_list': './list/train.txt',
                    'val_list': './list/val.txt'}

    ds = Cityscapes(data_setting, 'train', None)
    print(ds.__len__())
    print(16000 // 2975)
