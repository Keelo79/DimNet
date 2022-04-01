from model import DimNet
import torch
import numpy as np
import h5py
import cv2
import os
import time

net = DimNet(upscale_factor=4, device='cpu')
weight = torch.load('LF-DimNet_5x5_4xSR.pth.tar')
net.load_state_dict(weight['net'])
criterion_Loss = torch.nn.L1Loss()

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
        label = torch.from_numpy(label)
        size = label.size()
        label = label.reshape([1, size[0], size[1], size[2], size[3]])
        with torch.no_grad():
            start = time.time()
            data = net(data)
            end = time.time()
            print(end - start)

        print('%06d.h5' % (i + 1))
        print(criterion_Loss(data, label).data.cpu())