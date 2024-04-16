#import necessary libraries
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.nn as nn

class Disease_detection(nn.Module):
    def __init__(self):
        super(Disease_detection, self).__init__()
        # input shape: 3 * 224 * 224
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        # shape after conv1: 6 * 220 * 220
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # shape after pool: 6 * 110 * 110
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # shape after conv2: 16 * 106 * 106
        self.fc1 = nn.Linear(in_features=44944, out_features=120)
        # shape after fc1: 120
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        # shape after fc2: 84
        self.fc3 = nn.Linear(in_features=84, out_features=20)
        # shape after fc3: 20 with batch size 32
        self.dropout1 = nn.Dropout()
        self.dropout2 = nn.Dropout()
        # batch normalization layer
        self.batch_norm1 = nn.BatchNorm2d(6)
        self.batch_norm2 = nn.BatchNorm2d(16)
        self.softmax = nn.Softmax(dim=1)
        # explain batch normalization layer: https://towardsdatascience.com/batch-normalization-in-3-levels-of-understanding-14c2da90a338

    # Define the forward pass
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # batch normalization layer
        x = self.batch_norm1(x)
        x = self.pool(F.relu(self.conv2(x)))
        # batch normalization layer
        x = self.batch_norm2(x)
        x = x.view(-1, 16 * 53 * 53)
        # print(f"after view: {x.shape}")
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout1(x)
        x = self.fc3(x)
        x = self.dropout2(x)
        # print(f"after fc3: {x.shape}")
        x = self.softmax(x)
        # print(f"after log_softmax: {x.shape}")
        return x