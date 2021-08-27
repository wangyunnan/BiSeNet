import torch
import torch.nn.functional as F
import os.path as osp
import numpy as np
import sys
import cv2
import os
import logging
from tqdm import tqdm
from tensorboardX import SummaryWriter

from config import config
from datasets.cityscapes import Cityscapes
from dataloader import get_eval_loader
from network.bisenet import BiSeNet

#Cityscapes Class Name List
name_list = {0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence', 5: 'pole',
             6: 'traffic light', 7: 'traffic sign', 8: 'vegetation', 9: 'terrain',
             10: 'sky', 11: 'person', 12: 'rider', 13: 'car', 14: 'truck', 15: 'bus',
             16: 'train', 17: 'motorcycle', 18: 'bicycle'}

#Cityscapes Class Color Map (255 is change to 19)
color_map = {0: [128, 64, 128], 1: [244, 35, 232], 2: [70, 70, 70], 3: [102, 102, 156],
             4: [190, 153, 153], 5: [153, 153, 153], 6: [250, 170, 30], 7: [220, 220, 0],
             8: [107, 142, 35], 9: [152, 251, 152], 10: [70, 130, 180], 11: [220, 20, 60],
             12: [255, 0, 0], 13: [0, 0, 142], 14: [0, 0, 70], 15: [0, 60, 100], 16: [0, 80, 100],
             17: [0, 0, 230], 18: [119, 11, 32], 19: [0, 0, 0]}

def init():
    #Logger setting
    log_path = osp.join(config.save_root, 'log')
    if not osp.exists(log_path):
        os.makedirs(log_path)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(log_path, 'eval_log.txt'))
    ch = logging.StreamHandler()
    fh.setLevel(logging.DEBUG)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    #Writer setting
    data_path = osp.join(config.save_root, 'eval/board')
    writer = SummaryWriter(data_path)

    #Seed setting
    # cudnn.benchmark = True
    # torch.manual_seed(config.seed)
    # torch.cuda.manual_seed(config.seed)
    # random.seed(config.seed)
    # np.random.seed(config.seed)
    # torch.random.manual_seed(config.seed)
    # torch.backends.cudnn.benchmark = True

    return logger, writer

class Meature(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.hist = np.zeros((self.num_class, ) * 2, dtype=np.float32)

    def get_PA(self):
        PA = np.diag(self.hist).sum() / self.hist.sum() * 100
        PA = PA.round(4)
        return PA

    def get_CPAs(self):
        CPA = np.diag(self.hist) / self.hist.sum(axis=0)
        return CPA

    def get_MPA(self):
        CPA = self.get_CPAs()
        MPA = np.nanmean(CPA) * 100
        MPA = MPA.round(4)
        return MPA

    def get_IoUs(self):
        IoUs = np.diag(self.hist) / (np.sum(self.hist, axis=1) + np.sum(self.hist, axis=0) - np.diag(self.hist))
        return IoUs

    def get_MIoU(self):
        IoUs = self.get_IoUs()
        MIoU = np.nanmean(IoUs) * 100
        MIoU = MIoU.round(4)
        return MIoU

    def get_FWIoU(self):
        freq = np.sum(self.hist, axis=1) / np.sum(self.hist)
        IoU = np.diag(self.hist) / (np.sum(self.hist, axis=1) + np.sum(self.hist, axis=0) - np.diag(self.hist))
        FWIoU = (freq[freq > 0] * IoU[freq > 0]).sum() * 100
        FWIoU = FWIoU.round(4)
        return FWIoU

    def get_radios(self):
        radios = np.sum(self.hist, axis=1) / np.sum(self.hist)
        return radios

    def get_all_info(self):

        info_dict = {}
        CPAs = self.get_CPAs()
        IoUs = self.get_IoUs()
        radios = self.get_radios()
        for i in range(self.num_class):
            temp_dict = {}
            cpa = str(round(CPAs[i] * 100, 4)) if not np.isnan(CPAs[i]) else 'nan'
            iou = str(round(IoUs[i] * 100, 4)) if not np.isnan(IoUs[i]) else 'nan'
            radio = str(round(radios[i] * 100, 4)) if not np.isnan(radios[i]) else 'nan'
            temp_dict['iou'] = iou
            temp_dict['radio'] = radio
            temp_dict['cpa'] = cpa
            info_dict[name_list[i]] = temp_dict

        PA = self.get_PA()
        MPA = self.get_MPA()
        mIoU  = self.get_MIoU()
        FWIoU = self.get_FWIoU()

        return info_dict, PA, MPA, mIoU, FWIoU

    def generate_confusion_matrix(self, gts, preds):
        mask = (gts >= 0) &  (gts < self.num_class)
        index = gts[mask].astype('int') * self.num_class + preds[mask].astype('int')
        hist = np.bincount(index, minlength=self.num_class**2)
        confusion_matrix = hist.reshape(self.num_class, self.num_class)

        return confusion_matrix

    def add_batch(self, gts, preds):
        assert gts.shape == preds.shape
        self.hist += self.generate_confusion_matrix(gts, preds)

    def save_matrix(self, path):
        np.save(osp.join(path, "confusion_matrix"), self.hist)

    def read_matrix(self, path):
        self.hist = np.load(osp.join(path, "confusion_matrix.npy"))

    def reset(self):
        self.hist = np.zeros((self.num_class, ) * 2, dtype=np.float32)

class Evaluator(object):
    def __init__(self, ckpt_path='', logger=None, writer=None):
        # Checkpoint and log setting
        self.ckpt_path = ckpt_path
        self.logger = logger
        self.writer = writer

        # Image eval setting
        self.scales = config.eval_scale
        self.stride_rate = config.eval_stride_rate
        self.num_classes = config.num_classes
        self.crop_size = config.eval_crop_size
        self.flip = config.eval_flip

        # DataLoader
        self.eval_loader = get_eval_loader(Cityscapes)

        # Meature setting
        self.meature = Meature(self.num_classes)
        self.meature.reset()

        # Model setting
        self.model = BiSeNet(self.num_classes)
        self.model.load_state_dict(torch.load(self.ckpt_path, map_location='cuda:0'))
        self.logger.info('Checkpoint loaded successfully!')
        self.model.cuda()
        self.model.eval()


    def start(self):
        # Print information
        self.logger.info('Evaluating the model at {}'.format(self.ckpt_path))
        info_dict,  PA, MPA, mIoU, FWIoU = self.validate()

        # Write and show information
        self.logger.info('The results of the validation are as follows:')
        self.logger.info("Name\t\t| Radio(%)\t| CPA(%)\t| IoU(%)")
        self.logger.info("-" * 55)
        for name, child_dict in info_dict.items():
            self.logger.info("{}\t| {}\t| {}\t| {}".format(
                name.ljust(12), child_dict['radio'].ljust(8),
                child_dict['cpa'].ljust(8),child_dict['iou'].ljust(8)))
        self.logger.info("-" * 55)
        self.logger.info("Total PA\t: "    + str(PA))
        self.logger.info("Total MPA\t: "   + str(MPA))
        self.logger.info("Total FWIoU\t: " + str(FWIoU))
        self.logger.info("Total mIoU\t: "  + str(mIoU))

    def validate(self):
        self.logger.debug('Starting evaluation')

        with torch.no_grad():
            pbar = tqdm(self.eval_loader, file=sys.stdout)
            for idx, (imgs, gts) in enumerate(pbar):
                imgs = imgs.cuda()
                preds = self.scale_eval(imgs)
                #self.visualize(preds[0],gts.data.numpy()[0])
                self.meature.add_batch(gts.data.numpy(), preds)

        self.meature.save_matrix(osp.join(config.save_root, 'eval'))
        return self.meature.get_all_info()


    def scale_eval(self, imgs):
        N, C, H, W = imgs.shape
        probs = torch.zeros((N, self.num_classes, H, W))

        for sc in self.scales:
            sH, sW = [int(H * sc), int(W * sc)]
            imgs_scale = F.interpolate(imgs, (sH, sW), mode='bilinear', align_corners=True)
            prob = self.sliding_eval(imgs_scale)
            prob = F.interpolate(prob, (H, W), mode='bilinear', align_corners=True)
            probs += prob.detach().cpu()
        probs = probs.numpy()
        preds = np.argmax(probs, axis=1)

        return preds

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

        return prob

    def inference(self, imgs):
        out = self.model(imgs)[0]
        prob = F.softmax(out, 1)

        if self.flip:
            imgs = torch.flip(imgs, dims=(3,))
            out = self.model(imgs)[0]
            out = torch.flip(out, dims=(3,))
            prob += F.softmax(out, 1)

        prob = torch.exp(prob)
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

    def visualize(self, pred, gt):
        color_list = []
        color_num = len(color_map)
        for i in range(color_num):
            color_list.append(color_map[i])
        mask = (gt == 255)
        pred[mask] = color_num - 1
        color_list = np.asarray(color_list)
        show = color_list[pred.astype('int')]


        img = cv2.cvtColor(np.uint8(show), cv2.COLOR_RGB2BGR)
        cv2.imwrite("visual.png", img)

if __name__ == '__main__':
    # Processing init
    logger, writer = init()

    #Validation
    agent_eval = Evaluator(ckpt_path=osp.join('./save/2021_08_25_20_14_17/ckpt/epoch-428.pth'), logger=logger,writer=writer)
    agent_eval.start()

    #Close writer
    writer.close()