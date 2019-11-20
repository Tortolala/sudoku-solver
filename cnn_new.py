import torch
import torchvision
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os, os.path


class NumbersCNN(nn.Module):
    def __init__(self):
        super(NumbersCNN,self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.max1 = nn.MaxPool2d(2, 2)
        self.cnn2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1)
#         self.max2 = nn.MaxPool2d(2, 2)
        self.linear = nn.Linear(25*25*20,512)
#         self.linear2 = nn.Linear(1024,512)
        self.linear3 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
#         self.drop = nn.Dropout(p=0.3)
 
    def forward(self,x):
        n = x.size(0)
        x = self.relu(self.cnn1(x))
        x = self.relu(self.max1(x))
        x = self.relu(self.cnn2(x))
#         x = self.relu(self.max2(x))
        x = x.view(n,-1)
        x = self.linear(x)
#         x = self.drop(x)
#         x = self.linear2(x)
        x = self.linear3(x)
        return x

model = NumbersCNN()

checkpoint = torch.load("checkpoint_new", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])


def predict(img):
    img = np.array(img, dtype='f')
    padding = 22
    img = cv.copyMakeBorder(img, padding, padding, padding, padding, cv.BORDER_CONSTANT, value=255)
    img = cv.resize(img, dsize=(50,50), interpolation = cv.INTER_CUBIC)
    new_img = torch.tensor([[img]])
    x = model(new_img)
    _, pred = torch.max(x,1)
    return int(pred)
