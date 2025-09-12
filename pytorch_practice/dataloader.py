import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('logs')

test_data = torchvision.datasets.CIFAR10(
    root = './dataset',
    train = False,
    download = True,
    transform = transforms.ToTensor()
)

test_loader = DataLoader(
    dataset = test_data,
    batch_size = 64,
    shuffle = True,
    num_workers = 0,
    drop_last = True
)
for epoch in range(2):
    for step, (imgs, targets) in enumerate(test_loader):
        # print(img.shape)
        # print(target)
        writer.add_images('Epoch: {}'.format(epoch), imgs, step)
        step += 1

writer.close()