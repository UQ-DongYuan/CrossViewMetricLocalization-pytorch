from torchvision import models
import torch
import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self, keep_prob):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        features = list(vgg16.features.children())
        # conv 3-64
        self.layer1 = nn.Sequential(*features[:4])
        # conv 64-128
        self.layer2 = nn.Sequential(*features[5:9])
        # conv 128-256
        self.layer3 = nn.Sequential(*features[10:16])
        # conv 256-512
        self.layer4 = nn.Sequential(*features[17:19])
        # conv 512-512
        self.layer5 = nn.Sequential(*features[19:21])
        # conv 512-512
        self.layer6 = nn.Sequential(*features[21:23])
        # conv 512-512
        self.layer7 = nn.Sequential(*features[24:26])
        # conv 512-512
        self.layer8 = nn.Sequential(*features[26:28])
        # conv 512-512
        self.layer9 = nn.Sequential(*features[28:30])

        self.dropout = nn.Dropout(p=keep_prob)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

    def forward(self, x):
        layer1_output = self.layer1(x)
        layer1_pool_output = self.max_pool(layer1_output)

        layer2_output = self.layer2(layer1_pool_output)
        layer2_pool_output = self.max_pool(layer2_output)

        layer3_output = self.layer3(layer2_pool_output)
        layer3_pool_output = self.max_pool(layer3_output)

        layer4_output = self.layer4(layer3_pool_output)
        layer4_output = self.dropout(layer4_output)

        layer5_output = self.layer5(layer4_output)
        layer5_output = self.dropout(layer5_output)

        layer6_output = self.layer6(layer5_output)
        layer6_output = self.dropout(layer6_output)
        layer6_pool_output = self.max_pool(layer6_output)

        layer7_output = self.layer7(layer6_pool_output)
        layer7_output = self.dropout(layer7_output)

        layer8_output = self.layer8(layer7_output)
        layer8_output = self.dropout(layer8_output)

        layer9_output = self.layer9(layer8_output)
        layer9_output = self.dropout(layer9_output)

        return layer9_output, layer1_output, layer2_output, layer3_output, layer6_output

if __name__ == '__main__':
    vgg16 = VGG16(keep_prob=0.5)
    print(vgg16)


