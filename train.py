import torch
import torch.nn as nn
import torch.distributed as dist

import sys
import os
import os.path as osp
import argparse
import logging
from tqdm import tqdm
from tensorboardX import SummaryWriter

from config import config
from datasets.cityscapes import Cityscapes
from dataloader import get_train_loader
from network.bisenet import BiSeNet
from criterion import OhemCELoss
from evaluate import Evaluator, Meature
from optimizer import Optimizer

#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py

def init():
    # Logger setting
    log_path = osp.join(config.save_root, 'log')
    if not osp.exists(log_path):
        os.makedirs(log_path)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(log_path, 'train_log.txt'))
    ch = logging.StreamHandler()
    fh.setLevel(logging.DEBUG)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    # Writer setting
    data_path = osp.join(config.save_root, 'train/board')
    writer = SummaryWriter(data_path)

    return logger, writer

class Trainer(object):
    def __init__(self, logger=None, writer=None):
        # Log setting
        self.logger = logger
        self.writer = writer

        # Distributed training setting
        self.num_gpus = torch.cuda.device_count()
        args = self.parse_args()
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method='env://',
            world_size=torch.cuda.device_count(),
            rank=args.local_rank)

        # DataLoader setting
        self.train_loader, self.train_sampler =  get_train_loader(Cityscapes)

        # Criterion setting
        min_kept = int(config.batch_size * config.crop_size[0] * config.crop_size[1] // 16)
        self.criterion_out = OhemCELoss(thresh=0.7, min_kept=min_kept, ignore_lable=config.ignore_label)
        self.criterion_aux16 = OhemCELoss(thresh=0.7, min_kept=min_kept, ignore_lable=config.ignore_label)
        self.criterion_aux32 = OhemCELoss(thresh=0.7, min_kept=min_kept, ignore_lable=config.ignore_label)

        # Model setting
        self.num_classes = config.num_classes
        self.model = BiSeNet(self.num_classes)
        self.model.cuda()
        self.model.train()
        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[args.local_rank, ],
            output_device=args.local_rank
        )

        # Optimizer setting
        self.num_epochs = config.num_epochs
        self.num_iters = config.num_iters
        self.optimizer = Optimizer(
            params=self.model.module.get_params(config.lr),
            base_lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            total_iter=self.num_iters,
            lr_power=config.lr_power,
            warmup_steps=1000,
            warmup_start_lr=1e-5,
        )

        # Checkpoint and resume setting
        self.start_epoch = 0
        self.current_epoch = 0
        self.current_iter = 0
        self.ckpt_path = osp.join(config.save_root, 'ckpt')
        if not osp.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        self.last_path = osp.join(self.ckpt_path, 'epoch-last.pth')

    def start(self):
        # Print information
        if dist.get_rank() == 0:
            self.logger.info('Global configuration as follows:')
            for key, val in config.items():
                self.logger.info("{:16} {}".format(key, val))

        # Start to training
        self.train()

    def train(self):
        # Epochs during training
        for epoch in range(self.start_epoch, self.num_epochs):
            self.current_epoch = epoch
            self.train_one_epoch()

            # Save checkpoint
            if ((epoch > config.num_epochs - 20) or (epoch % config.snapshot_epoch == 0)) and (dist.get_rank()):
                save_path = osp.join(self.ckpt_path, 'epoch-{}.pth'.format(epoch))
                state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
                torch.save(state, save_path)
                self.logger.info('Saving checkpoint to {}'.format(save_path))
                if osp.isdir(self.last_path) or osp.isfile(self.last_path) or osp.islink(self.last_path):
                    os.remove(self.last_path)
                os.system('ln -s {} {}'.format(save_path, self.last_path))

    def train_one_epoch(self):
        self.train_sampler.set_epoch(self.current_epoch)
        pbar = tqdm(self.train_loader, file=sys.stdout)
        for idx, (imgs, gts) in enumerate(pbar):
            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)

            # Forward
            self.optimizer.zero_grad()
            out, aux16, aux32 = self.model(imgs)
            loss_out = self.criterion_out(out, gts)
            loss_aux16 = self.criterion_aux16(aux16, gts)
            loss_aux32 = self.criterion_aux32(aux32, gts)
            loss = loss_out + loss_aux16 + loss_aux32

            # Backward
            loss.backward()
            self.optimizer.step()

            # Print information and save log file
            reduce_loss = self.reduce_tensor(loss)
            self.current_iter += 1
            print_str = 'Epoch-{}/{}'.format(self.current_epoch + 1, self.num_epochs).ljust(12) \
                        + 'Iter-{}/{}'.format(self.current_iter, self.num_iters).ljust(12) \
                        + 'lr=%.2e ' % self.optimizer.lr \
                        + 'loss=%.4f' % reduce_loss
            pbar.set_description(print_str, refresh=False)
            if (dist.get_rank() == 0) and self.current_iter % config.log_iter == 0:
                self.logger.debug(print_str)
                self.writer.add_scalar('train/learning_rate', self.optimizer.lr, self.current_iter)
                self.writer.add_scalar('train/loss', reduce_loss, self.current_iter)

            # Stop training
            if self.current_iter == self.num_iters:
                break

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
        args = parser.parse_args()
        return args

    def reduce_tensor(self, tensor):
        reduce_tensor = tensor.clone()
        dist.all_reduce(reduce_tensor, dist.ReduceOp.SUM)
        reduce_tensor.div_(self.num_gpus)
        return reduce_tensor.item()

if __name__ == '__main__':
    # Processing init
    logger, writer = init()

    #Train
    agent_train = Trainer(logger=logger, writer=writer)
    agent_train.start()

    #Validation
    #agent_eval = Evaluator(ckpt_path=osp.join(config.save_root, 'ckpt/epoch-79.pth'), logger=logger, writer=writer)
    #agent_eval.start()

    #Close writer
    writer.close()