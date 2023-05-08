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

    def forward(self, x, sat512, sat256, sat128, sat64, sat32):
        deconv1 = self.deconv1(x, output_size=(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3]*2))
        conv1_1 = self.conv1_1(deconv1)
        conv1_2 = self.conv1_2(conv1_1)

        deconv2 = self.deconv2(conv1_2, output_size=(conv1_2.shape[0], conv1_2.shape[1], conv1_2.shape[2]*2, conv1_2.shape[3]*2))
        conv2_1 = self.conv2_1(torch.cat([deconv2, sat32], dim=1))
        conv2_2 = self.conv2_2(conv2_1)

        deconv3 = self.deconv3(conv2_2, output_size=(conv2_2.shape[0], conv2_2.shape[1], conv2_2.shape[2]*2, conv2_2.shape[3]*2))
        conv3_1 = self.conv3_1(torch.cat([deconv3, sat64], dim=1))
        conv3_2 = self.conv3_2(conv3_1)

        deconv4 = self.deconv4(conv3_2, output_size=(conv3_2.shape[0], conv3_2.shape[1], conv3_2.shape[2]*2, conv3_2.shape[3]*2))
        conv4_1 = self.conv4_1(torch.cat([deconv4, sat128], dim=1))
        conv4_2 = self.conv4_2(conv4_1)

        deconv5 = self.deconv5(conv4_2, output_size=(conv4_2.shape[0], conv4_2.shape[1], conv4_2.shape[2]*2, conv4_2.shape[3]*2))
        conv5_1 = self.conv5_1(torch.cat([deconv5, sat256], dim=1))
        conv5_2 = self.conv5_2(conv5_1)

        deconv6 = self.deconv6(conv5_2, output_size=(conv5_2.shape[0], conv5_2.shape[1], conv5_2.shape[2]*2, conv5_2.shape[3]*2))
        conv6_1 = self.conv6_1(torch.cat([deconv6, sat512], dim=1))
        conv6_2 = self.conv6_2(conv6_1)

        return conv6_2



