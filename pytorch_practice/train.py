import torchvision
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#Load Data

train_data = torchvision.datasets.CIFAR10(
    root = '/dataset',
    train = True,
    download = True,
    transform = torchvision.transforms.ToTensor()
)

test_data = torchvision.datasets.CIFAR10(
    root = '/dataset',
    train = False,
    download = True,
    transform = torchvision.transforms.ToTensor()
)

train_data_size = len(train_data)
test_data_size = len(test_data)

train_dataloader = DataLoader(train_data, batch_size = 64)
test_dataloader = DataLoader(test_data, batch_size = 64)

#Create a neural network

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding = 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding = 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding = 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x

model = NeuralNetwork()
model = model.cuda()

#Loss function and Optimizer

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

writer = SummaryWriter('logs')

#Train the model

epoch = 20
total_train_step = 0
total_test_step = 0

for i in range(epoch):
    print('Epoch {}'.format(i+1))
    total_train_loss = 0

    model.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = model(imgs)

        loss = loss_fn(outputs, targets)
        total_train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1

        if total_train_step % 100 == 0:
            writer.add_scalar('train_loss1', loss.item(), total_train_step)

    print('Total loss in epoch {}: {}'.format(i+1, total_train_loss))

    # Validation

    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():

        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    total_accuracy = total_accuracy / test_data_size

    print('Total loss in test set: {}'.format(total_test_loss))
    print('Total accuracy in test set: {}'.format(total_accuracy))

    writer.add_scalar('test loss1', total_test_loss, total_test_step)
    writer.add_scalar('test accurary', total_accuracy, total_test_step)

    total_test_step += 1

    if i == epoch - 1:
        torch.save(model, 'model.pth')
        print('model saved')

writer.close()