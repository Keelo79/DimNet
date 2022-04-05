import os

import numpy as np

from model import DimNet
import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
import h5py
from torch.autograd import Variable
from test import test


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--upscale_factor", type=int, default=4, help="upscale factor")

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-4, help='initial learning rate')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=25, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decaying factor')

    parser.add_argument('--trainset_dir', type=str, default='./data')
    parser.add_argument('--model_name', type=str, default='LF-DimNet_5x5_4xSR.pth.tar')
    parser.add_argument('--load_pretrain', type=bool, default=True)

    parser.add_argument('--num_works', type=int, default=1)

    return parser.parse_args()


cfg = parse_args()
net = DimNet(cfg.upscale_factor, cfg.device)


def load(model_name):
    weight = torch.load(model_name)
    net.load_state_dict(weight['net'])
    best = weight['best']
    return best


def save(model_name, best):
    state = {'net': net.state_dict(), 'best': best}
    torch.save(state, model_name)


class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir):
        super(TrainSetLoader, self).__init__()
        self.dataset_dir = dataset_dir
        file_list = os.listdir(dataset_dir)
        item_num = len(file_list)
        self.item_num = item_num

    def __getitem__(self, index):
        dataset_dir = self.dataset_dir
        index = index + 1
        file_name = [dataset_dir + '/%06d' % index + '.h5']
        with h5py.File(file_name[0], 'r') as hf:
            data = np.array(hf.get('data'))
            label = np.array(hf.get('label'))

            data = data.transpose((3, 2, 1, 0))
            label = label.transpose((3, 2, 1, 0))

            data = torch.from_numpy(data)
            label = torch.from_numpy(label)

        return data, label

    def __len__(self):
        return self.item_num


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.fill_(0.0)
        # torch.nn.init.zeros_(m.weight.data)
        # torch.nn.init.kaiming_uniform_(m.weight.data)


if __name__ == '__main__':
    net.apply(weights_init_xavier)
    if cfg.load_pretrain:
        best = load(cfg.model_name)
    else:
        best = 1
    train_set = TrainSetLoader(cfg.trainset_dir)
    train_loader = DataLoader(dataset=train_set,
                              num_workers=cfg.num_works,
                              batch_size=cfg.batch_size,
                              shuffle=True)
    criterion_Loss = torch.nn.L1Loss().to(cfg.device)
    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)

    for idx_epoch in range(cfg.n_epochs):
        for idx_iter, (data, label) in enumerate(train_loader,1):
            data, label = Variable(data).to(cfg.device), Variable(label).to(cfg.device)
            out = net(data)
            loss = criterion_Loss(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        loss_ave,time_ave=test(net)
        print(idx_epoch,loss_ave,best)
        if loss_ave < best:
            best = loss_ave
            save(cfg.model_name, best)
