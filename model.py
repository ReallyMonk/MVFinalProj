import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F


class LeNet5(nn.Module):
    def __init__(self, class_num, is_gray_scale):
        super(LeNet5, self).__init__()

        if is_gray_scale:
            input_channel = 1
        else:
            input_channel = 3

        self.conv1 = nn.Conv2d(input_channel, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, class_num)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResBlock, self).__init__()
        self.basic = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel),
        )
        self.shortcut = shortcut

    def forward(self, x):
        out = self.basic(x)
        residual = x if self.shortcut is None else self.shortcut(x)
        out += residual
        return nn.ReLU(inplace=True)(out)


class ResNet34(nn.Module):
    def __init__(self, class_num, is_gray_scale):
        super(ResNet34, self).__init__()
        if is_gray_scale:
            input_channel = 1
        else:
            input_channel = 3

        self.pre = nn.Sequential(
            nn.Conv2d(input_channel, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        self.layer1 = self.__make_layer__(64, 128, 3)
        self.layer2 = self.__make_layer__(128, 256, 4, stride=2)
        self.layer3 = self.__make_layer__(256, 512, 6, stride=2)
        self.layer4 = self.__make_layer__(512, 512, 3, stride=2)
        self.fc = nn.Linear(512, class_num)

    def __make_layer__(self, inchannel, outchannel, block_num, stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel),
        )
        layer = []
        layer.append(ResBlock(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layer.append(ResBlock(outchannel, outchannel))

        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        return self.fc(x)
