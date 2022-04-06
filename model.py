import torch
import torch.nn as nn
from convNd import convNd


class DimNet(nn.Module):
    def __init__(self, upscale_factor, device='cpu'):
        super(DimNet, self).__init__()
        self.upscale_factor = upscale_factor
        self.device = device
        self.padder = Padder(self.device).to(self.device)
        self.dim_group = DimGroup(self.upscale_factor).to(self.device)
        self.bottle_neck = BottleNeck().to(self.device)
        self.recon_block = ReconBlock(self.upscale_factor, self.device).to(self.device)

    def forward(self, pic):
        pic = self.padder(pic)
        pic1, pic2 = self.dim_group(pic)
        pic = self.bottle_neck(pic1, pic2)
        pic = self.recon_block(pic)
        return pic


class Padder(nn.Module):
    def __init__(self, device):
        super(Padder, self).__init__()
        self.device = device

    def forward(self, pic_in):
        padded_size = list(pic_in.size())
        padded_size[3] += 8
        padded_size[4] += 8
        out = torch.ones(padded_size).to(self.device) / 2
        out[:, :, :, 4:padded_size[3] - 4, 4:padded_size[4] - 4] = pic_in
        return out


class DimGroup(nn.Module):
    def __init__(self, upscale_factor):
        super(DimGroup, self).__init__()
        self.upscale_factor = upscale_factor
        self.dim_block_1 = DimBlock_1(self.upscale_factor)
        self.dim_block_2 = DimBlock_2(self.upscale_factor)

    def forward(self, pic_in):
        pic_out_1, pic_out_2 = self.dim_block_1(pic_in), self.dim_block_2(pic_in)
        return pic_out_1, pic_out_2


class DimBlock_1(nn.Module):
    def __init__(self, upscale_factor):
        super(DimBlock_1, self).__init__()
        self.upscale_factor = upscale_factor
        self.core = convNd(in_channels=1,
                           out_channels=self.upscale_factor ** 2 * 5 ** 2,
                           num_dims=4,
                           kernel_size=(5, 5, 9, 9),
                           stride=1,
                           padding=0)
        self.ReLu = nn.ReLU(inplace=True)

    def forward(self, pic_in):
        size = pic_in.size()
        pic_in = pic_in.reshape([size[0], 1, size[1], size[2], size[3], size[4]])
        pic_out = self.core(pic_in)
        return pic_out


class DimBlock_2(nn.Module):
    def __init__(self, upscale_facotr):
        super(DimBlock_2, self).__init__()
        self.upscale_factor = upscale_facotr
        self.core_1 = convNd(in_channels=1,
                             out_channels=self.upscale_factor * 5,
                             num_dims=4,
                             kernel_size=(3, 3, 5, 5),
                             stride=1,
                             padding=0)
        self.core_2 = convNd(in_channels=self.upscale_factor * 5,
                             out_channels=self.upscale_factor ** 2 * 5 ** 2,
                             num_dims=4,
                             kernel_size=(3, 3, 5, 5),
                             stride=1,
                             padding=0)
        self.ReLu = nn.ReLU(inplace=True)

    def forward(self, pic_in):
        size = pic_in.size()
        pic_in=pic_in.reshape([size[0],1,size[1],size[2],size[3],size[4]])
        buffer=self.ReLu(self.core_1(pic_in))
        pic_out=self.ReLu(self.core_2(buffer))
        return pic_out


class BottleNeck(nn.Module):
    def __init__(self):
        super(BottleNeck, self).__init__()

    def forward(self, pic_in_1, pic_in_2):
        return (pic_in_1 + pic_in_2) / 2


class ReconBlock(nn.Module):
    def __init__(self, upscale_factor, device):
        super(ReconBlock, self).__init__()
        self.device = device
        self.upscale_factor = upscale_factor

    def forward(self, pic_in):
        size = pic_in.size()
        out = torch.zeros([size[0], 5, 5,
                           size[4] * self.upscale_factor,
                           size[5] * self.upscale_factor]).to(self.device)
        for k in range(size[0]):
            buffer = torch.zeros([1, size[1], size[4], size[5]]).to(self.device)
            buffer[0, :, :, :] = pic_in[k, :, 0, 0, :, :]
            upscaled = torch.pixel_shuffle(buffer, 4)
            for i in range(0, 5):
                out[k, i, :, :, :] = upscaled[0, i * 5:i * 5 + 5, :, :]
        return out


if __name__ == '__main__':
    data = torch.ones([2, 5, 5, 80, 80]).cuda()
    net = DimNet(upscale_factor=4, device='cuda:0')
    loss=torch.nn.L1Loss()
    with torch.no_grad():
        out = net(data)
    print(out.size())
