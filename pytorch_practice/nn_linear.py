import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(
    root = '/dataset',
    train = False,
    download = True,
    transform = torchvision.transforms.ToTensor()
)

dataloader = DataLoader(dataset, batch_size = 64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output

model = NeuralNetwork()

for step, (imgs, target) in enumerate(dataloader):
    print(imgs.shape)
    output = torch.flatten(imgs)
    print(output.shape)
    output = model(output)
    print(output.shape)
