import os
import os.path as osp
import cv2
import numpy as np

lb_map = {0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0,
          8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255,
          16: 255, 17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10,
          24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255,
          31: 16, 32: 17, 33: 18, -1: -1}

def convert_labels(label):
    for k, v in lb_map.items():
        label[label == k] = v
    return label


if __name__ == '__main__':
      root = '/media/wangyunnan/Data/datasets/Cityscapes'
      list_path = '/media/wangyunnan/Data/datasets/Cityscapes/list/labelIds/trainval.txt'

      with open(list_path, 'r') as fr:
          files = fr.readlines()

      file_names = []
      for item in files:
          _, gt_name = item.strip().split('\t')
          gt_path =  osp.join(root, gt_name)
          gt = np.array(cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE))
          new_gt = convert_labels(gt)
          cv2.imwrite(gt_path.replace('_labelIds.png','_trainIds.png'), new_gt)


