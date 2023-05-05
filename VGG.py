from torchvision import models
import torch
import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        layer1 = vgg16.features


if __name__ == '__main__':
    vgg = models.vgg16(pretrained=True)
    features = list(vgg.features.children())
    enc1 = nn.Sequential(*features[3:7])
    print(enc1)


