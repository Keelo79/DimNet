from model import DimNet
import torch
import numpy as np
import h5py
import cv2
import os
import time

net = DimNet(upscale_factor=4, device='cuda:0')
weight = torch.load('LF-DimNet_5x5_4xSR.pth.tar')
net.load_state_dict(weight['net'])

def test(net):
    ave_score = []
    ave_time = []
    criterion_Loss = torch.nn.L1Loss().to('cuda:0')

    file_list = os.listdir('./data')
    item_num = len(file_list)

    for i in range(item_num):
        with h5py.File('./data/%06d.h5' % (i + 1), 'r') as hf:
            data = np.array(hf.get('data'))
            label = np.array(hf.get('label'))
            data = data.transpose((3, 2, 1, 0))
            label = label.transpose((3, 2, 1, 0))
            data = torch.from_numpy(data)
            size = data.size()
            data = data.reshape([1, size[0], size[1], size[2], size[3]])
            label = torch.from_numpy(label).to('cuda:0')
            size = label.size()
            label = label.reshape([1, size[0], size[1], size[2], size[3]])
            with torch.no_grad():
                start = time.time()
                data = net(data)
                end = time.time()
                # print(end - start)
                ave_time.append(end - start)

            # print('%06d.h5' % (i + 1), criterion_Loss(data, label).data.cpu())
            ave_score.append(criterion_Loss(data, label).data.cpu())
    loss_ave = np.array(ave_score).mean()
    time_ave = np.array(ave_time).mean()
    return loss_ave, time_ave


if __name__ == '__main__':
    loss_ave, time_ave = test(net)
    print('average_score=', loss_ave)
    print('average_time=', time_ave)
