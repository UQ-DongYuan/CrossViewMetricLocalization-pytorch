import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, in_channels=4097, out_channels=1024):
        super(Decoder, self).__init__()
        # deconv1 and conv1, height, width: 16*16
        self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.conv1_1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        # deconv2 and conv2, height, width: 32*32
        self.deconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv2_1 = nn.Conv2d(in_channels=768, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        # deconv3 and conv3, height, width: 64*64
        self.deconv3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv3_1 = nn.Conv2d(in_channels=640, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        # deconv4 and conv4, height, width: 128*128
        self.deconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv4_1 = nn.Conv2d(in_channels=320, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        # deconv5 and conv5, height, width: 256*256
        self.deconv5 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv5_1 = nn.Conv2d(in_channels=160, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        # deconv6 and conv6, height, width: 512*512
        self.deconv6 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv6_1 = nn.Conv2d(in_channels=80, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)



