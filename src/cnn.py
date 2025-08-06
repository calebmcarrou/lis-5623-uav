#######################################################################################
# cnn.py
#######################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

# this is heavily lifted from the pytorch tutorial on CNNs
# modified to take in the 32x32 resized images in the UAV dataset
# and to have 1 class, not 10
class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # fully connected layers for 1 class
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # for 32x32
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1) # 1 class prediction

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # no sigmoid for final layer
        
        return x
