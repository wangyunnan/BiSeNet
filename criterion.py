import torch
import torch.nn as nn

class OhemCELoss(nn.Module):
    def __init__(self, thresh, min_kept, ignore_lable=255):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.min_kept = min_kept
        self.ignore_lb = ignore_lable
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_lable, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criterion(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.min_kept] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.min_kept]
        return torch.mean(loss)

if __name__ == '__main__':
    loss1 = OhemCELoss(thresh=0, min_kept=4, )
    loss3 = nn.CrossEntropyLoss()
    x = torch.randn(8,19,1024,1024)
    label = torch.empty([8,1024,1024], dtype=torch.long).random_(19)
    out1 = loss1(x,label)

    print(out1)