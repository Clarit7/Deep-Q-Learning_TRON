import torch.nn as nn
import torch.nn.functional as F
from config import *


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gamma = GAMMA

        self.conv1=nn.Conv2d(1, 32, 6)
        self.conv2 = nn.Conv2d(32, 64, 3)

        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout=nn.Dropout(p=0.2)

        self.relu=nn.ReLU()

        self.fc1 = nn.Linear(64*5*5, 256)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(-1, 64 * 5 * 5)

        x=self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def act(self, x):
        output = self(x)
        return torch.argmax(output, dim=1)

