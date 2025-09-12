import torch
import torchvision.transforms
from torch import nn
from torch.nn import Sigmoid
from torch.nn import ReLU
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                     [-1, 3]],
                     dtype = torch.float32)

input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)

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
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output

model = NeuralNetwork()

writer = SummaryWriter('logs')

for step, (imgs, target) in enumerate(dataloader):
    writer.add_images('sigmoid_input', imgs, step + 1)
    output = model(imgs)
    writer.add_images('sigmoid_output', output, step + 1)

writer.close()