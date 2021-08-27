import torch

class Optimizer(object):
    def __init__(self,
                params,
                base_lr,
                momentum,
                weight_decay,
                total_iter,
                lr_power,
                warmup_steps,
                warmup_start_lr,
                *args, **kwargs):

        self.base_lr = base_lr
        self.lr = self.base_lr
        self.total_iters = float(total_iter)
        self.lr_power = lr_power
        self.iter = 0.0

        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.warmup_factor = (self.base_lr / self.warmup_start_lr) ** (1. / self.warmup_steps)

        self.optim = torch.optim.SGD(
                params,
                lr = self.lr,
                momentum = momentum,
                weight_decay = weight_decay
        )

    def get_lr(self):
        if self.iter < self.warmup_steps:
            lr = self.warmup_start_lr * (self.warmup_factor ** self.iter)
        else:
            factor = (1 - (self.iter - self.warmup_steps) / (self.total_iters - self.warmup_steps)) ** self.lr_power
            lr = self.base_lr * factor
        return lr

    def step(self):
        self.lr = self.get_lr()
        for i in range(2):
            self.optim.param_groups[i]['lr'] = self.lr
        for i in range(2, len(self.optim.param_groups)):
            self.optim.param_groups[i]['lr'] = self.lr * 10
        self.iter += 1
        self.optim.step()

    def zero_grad(self):
        self.optim.zero_grad()

