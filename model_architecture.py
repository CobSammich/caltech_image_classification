"""
Definition of CNN model architecture in pytorch
"""

from typing import Tuple
import ipdb

import torch
import torch.nn.functional as F

class Conv_Block(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Conv_Block, self).__init__()

        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=(1, 1))
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.pool = torch.nn.MaxPool2d(kernel_size=(2,2))

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.pool(x)
        return x

class My_Model(torch.nn.Module):
    def __init__(self, n_classes: int) -> None:
        super(My_Model, self).__init__()

        self.block1 = Conv_Block(3, 32)      # 256 -> 128
        self.block2 = Conv_Block(32, 64)     # 128 ->  64
        self.block3 = Conv_Block(64, 128)    #  64 ->  32
        self.block4 = Conv_Block(128, 256)   #  32 ->  16
        self.block5 = Conv_Block(256, 512)   #  16 ->   8

        self.fc1 = torch.nn.Linear(int(8*8*512), 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.fc3 = torch.nn.Linear(512, 256)
        self.fc4 = torch.nn.Linear(256, 128)
        self.fc5 = torch.nn.Linear(128, int(n_classes))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch size
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        # images mutually exclusive - probabilities sum to 1
        x = F.softmax(x, dim=1)
        return x

if __name__ == "__main__":
    b = Conv_Block(3, 32)

    model = My_Model(10)
    ipdb.set_trace()


