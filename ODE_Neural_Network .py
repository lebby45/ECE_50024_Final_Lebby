import numpy as np 
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets,transforms

from torchdiffeq import odeint_adjoint as odeint

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
        self.conv= nn.Conv2d(32, 32, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.norm =nn.GroupNorm(min(16, 32), 32)


    def forward(self, t, x):
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(t,x)   
        x = self.norm(x)
        return x

# Define ODE Layers
class ODEBlock(nn.Module):

    def __init__(self, cnn):
        super(ODEBlock, self).__init__()
        self.cnn = cnn
        self.t = torch.tensor([0, 1]).float()

    def forward(self, y0):
        self.t = self.t.type_as(y0)
        out = odeint(self.cnn, y0, self.t, rtol=1e-3, atol=1e-3)
        return out[1]
    
class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)
    
input_layers = [nn.Conv2d(3, 32, 3, 1), nn.GroupNorm(min(16, 32), 32), nn.ReLU(inplace=True), nn.Conv2d(32, 32, 4, 2, 1),]

ode_layers = [ODEBlock(CNNet())]

output_layers = [nn.GroupNorm(min(16, 32), 32), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(32, 10)]

net = nn.Sequential(*input_layers,*ode_layers,*output_layers)

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

        running_loss += loss.item()
        if i % 2000 == 1999:    
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

# Save model
PATH = './ode_net.pth'
torch.save(net.state_dict(), PATH)

# Test Model
dataiter = iter(testloader)
images, labels = next(dataiter)

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

print(f'Accuracy of the ODE network: {100 * correct / total} %')