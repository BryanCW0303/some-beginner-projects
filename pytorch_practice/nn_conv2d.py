import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter

import nn_module

dataset = torchvision.datasets.CIFAR10(
    root = '/dataset',
    download = True,
    train = False,
    transform = torchvision.transforms.ToTensor()
)

dataloader = DataLoader(
    dataset,
    batch_size = 64
)

class Neural_Network(nn.Module):
    def __init__(self):
        super(Neural_Network, self).__init__()
        self.conv1 = Conv2d(
            in_channels=3,
            out_channels=6,
            kernel_size=3,
            stride=1,
            padding=0
        )
        self.conv2 = Conv2d(
            in_channels=6,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=0
        )

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)

model1 = Neural_Network()

writer = SummaryWriter('logs')

for step, (imgs, targets) in enumerate(dataloader):
    output = model1(imgs)
    if step == 0:
        print(imgs.shape)
        print(output.shape)
    writer.add_images('Conv2d_intput', imgs, step + 1)
    writer.add_images('Conv2d_output', output, step + 1)

writer.close()

# VGC16
