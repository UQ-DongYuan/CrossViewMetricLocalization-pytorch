from torchvision import models
import torch
import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self, keep_prob=0.8):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        features = list(vgg16.features.children())
        # conv 3-64
        self.layer1 = nn.Sequential(*features[:4])
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        # conv 64-128
        self.layer2 = nn.Sequential(*features[5:9])
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        # conv 128-256
        self.layer3 = nn.Sequential(*features[10:16])
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        # conv 256-512
        self.layer4 = nn.Sequential(*features[17:19])
        self.dropout_1 = nn.Dropout(p=keep_prob)

        # conv 512-512
        self.layer5 = nn.Sequential(*features[19:21])
        self.dropout_2 = nn.Dropout(p=keep_prob)

        # conv 512-512
        self.layer6 = nn.Sequential(*features[21:23])
        self.dropout_3 = nn.Dropout(p=keep_prob)
        self.max_pool_4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        # conv 512-512
        self.layer7 = nn.Sequential(*features[24:26])
        self.dropout_4 = nn.Dropout(p=keep_prob)

        # conv 512-512
        self.layer8 = nn.Sequential(*features[26:28])
        self.dropout_5 = nn.Dropout(p=keep_prob)

        # conv 512-512
        self.layer9 = nn.Sequential(*features[28:30])
        self.dropout_6 = nn.Dropout(p=keep_prob)

    def forward(self, x):
        layer1_output = self.layer1(x)
        layer1_pool_output = self.max_pool_1(layer1_output)

        layer2_output = self.layer2(layer1_pool_output)
        layer2_pool_output = self.max_pool_2(layer2_output)

        layer3_output = self.layer3(layer2_pool_output)
        layer3_pool_output = self.max_pool_3(layer3_output)

        layer4_output = self.layer4(layer3_pool_output)
        layer4_output = self.dropout_1(layer4_output)

        layer5_output = self.layer5(layer4_output)
        layer5_output = self.dropout_2(layer5_output)

        layer6_output = self.layer6(layer5_output)
        layer6_output = self.dropout_3(layer6_output)
        layer6_pool_output = self.max_pool_4(layer6_output)

        layer7_output = self.layer7(layer6_pool_output)
        layer7_output = self.dropout_4(layer7_output)

        layer8_output = self.layer8(layer7_output)
        layer8_output = self.dropout_5(layer8_output)

        layer9_output = self.layer9(layer8_output)
        layer9_output = self.dropout_6(layer9_output)

        return layer9_output, layer1_output, layer2_output, layer3_output, layer6_output


if __name__ == '__main__':
    vgg16 = VGG16(keep_prob=0.5)
    img = torch.randn(1, 3, 512, 512)
    # print(vgg16)
    result = vgg16(img)
    print(result[0].shape)
    print(result[1].shape)
    print(result[2].shape)
    print(result[3].shape)
    print(result[4].shape)

