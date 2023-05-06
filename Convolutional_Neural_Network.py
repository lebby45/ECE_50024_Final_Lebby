import numpy as np 
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets,transforms
import torch.optim as optim

# Load and normalize CIFAR10
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

# Define CNN
class CNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv365 = nn.Conv2d(3, 6, 5)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv6165 = nn.Conv2d(6, 16, 5)
        self.dense1 = nn.Linear(16 * 5 * 5, 128)
        self.dense2 = nn.Linear(128, 32)
        self.dense3 = nn.Linear(32, 10)


    def forward(self, x):
        x = self.maxpool(F.relu(self.conv365(x)))
        x = self.maxpool(F.relu(self.conv6165(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x

net = CNNet()

# Define Loss and optimizer

loss_fcn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


# Train network
epochs = 2
for epoch in range(epochs):  
    running_loss = 0.0
    for i, data in enumerate(trainloader,0):

        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = loss_fcn(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0


# Save model
PATH = './cnn_model.pth'
torch.save(net.state_dict(), PATH)

# Test Model
dataiter = iter(testloader)
images, labels = next(dataiter)

net = CNNet()
net.load_state_dict(torch.load(PATH))

outputs = net(images)

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data

        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the classic CNN network: {100 * correct / total} %')