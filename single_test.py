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

with h5py.File('./data/000006.h5', 'r') as hf:
    data = np.array(hf.get('data'))
    label = np.array(hf.get('label'))
    data = data.transpose((3, 2, 1, 0))
    label = label.transpose((3, 2, 1, 0))

    show = data[0, 0, :, :]
    show = show.squeeze()
    show = show * 255
    show = np.array(show, dtype='uint8')
    cv2.imshow('data', show)

    show = label[0, 0, :, :]
    show = show.squeeze()
    show = show * 255
    show = np.array(show, dtype='uint8')
    cv2.imshow('label', show)

    with torch.no_grad():
        data=torch.from_numpy(data)
        size = data.size()
        data = data.reshape([1, size[0], size[1], size[2], size[3]])
        show = net(data)
    show = show[0, 0, 0, :, :]
    show = show.squeeze()
    show = show * 255
    show = np.array(show, dtype='uint8')
    cv2.imshow('sr', show)

    cv2.waitKey()
