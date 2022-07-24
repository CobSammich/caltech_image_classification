"""
Definition of CNN model architecture in pytorch
"""

import ipdb

import torch
import torch.nn.functional as F

class Conv_Block(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Conv_Block, self).__init__()

        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=(1, 1))
        self.relu = torch.nn.ReLU(inplace=True)
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        self.pool = torch.nn.MaxPool2d(kernel_size=(2,2))

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
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

# ==================

class VGG_Conv_Block(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(VGG_Conv_Block, self).__init__()

        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=(1, 1))
        self.relu = torch.nn.ReLU(inplace=True)
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        #self.pool = torch.nn.MaxPool2d(kernel_size=(2,2))

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x

class VGG_Model(torch.nn.Module):
    def __init__(self, n_classes: int) -> None:
        super(VGG_Model, self).__init__()

        self.maxpool = torch.nn.MaxPool2d(kernel_size=(2,2), stride=2)

        self.block1_1 = VGG_Conv_Block(3, 64)      # 256 -> 128
        self.block1_2 = VGG_Conv_Block(64, 64)

        self.block2_1 = VGG_Conv_Block(64, 128)     # 128 ->  64
        self.block2_2 = VGG_Conv_Block(128, 128)

        self.block3_1 = VGG_Conv_Block(128, 256)    #  64 ->  32
        self.block3_2 = VGG_Conv_Block(256, 256)

        self.block4_1 = VGG_Conv_Block(256, 512)   #  32 ->  16
        self.block4_2 = VGG_Conv_Block(512, 512)

        self.block5_1 = VGG_Conv_Block(512, 512)   #  16 ->   8
        self.block5_2 = VGG_Conv_Block(512, 512)

        self.fc1 = torch.nn.Linear(int(8*8*512), 4096)
        self.fc2 = torch.nn.Linear(4096, 4096)
        self.fc3 = torch.nn.Linear(4096, 1000)
        self.fc4 = torch.nn.Linear(1000, int(n_classes))

    def forward(self, x):
        x = self.block1_1(x)
        x = self.block1_2(x)
        x = self.maxpool(x)

        x = self.block2_1(x)
        x = self.block2_2(x)
        x = self.maxpool(x)

        x = self.block3_1(x)
        x = self.block3_2(x)
        x = self.maxpool(x)

        x = self.block4_1(x)
        x = self.block4_2(x)
        x = self.maxpool(x)

        x = self.block5_1(x)
        x = self.block5_2(x)
        x = self.maxpool(x)

        x = torch.flatten(x, 1) # flatten all dimensions except batch size

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        # images mutually exclusive - probabilities sum to 1
        x = F.softmax(x, dim=1)
        return x

if __name__ == "__main__":
    b = Conv_Block(3, 32)

    model = My_Model(10)
    ipdb.set_trace()
