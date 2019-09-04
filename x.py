import torch
import torch.nn as nn
class NN(nn.Module):
  def __init__(self):
    super(NN,self).__init__()
    
    # 126
    self.conv1 = nn.Conv2d(8,32,(1,5))
    # 122
    self.conv2 = nn.Conv2d(32,64,(1,5))
    # 118
    self.m1 = nn.MaxPool2d((2,2))
    # 59
    self.conv3 = nn.Conv2d(64,128,(1,5))
    # 55
    self.conv4 = nn.Conv2d(128,64,(1, 5))
    # 51
    self.lin1 = nn.Linear(7040,200)
    self.lin2 = nn.Linear(200,30)
    self.lin3 = nn.Linear(30,3)
    
    self.relu = nn.ReLU()
    self.softmax = nn.Softmax()
  
  def forward(self,x):
    x = self.relu(self.conv1(x))
    x = self.relu(self.conv2(x))
    x = self.relu(self.conv3(x))
    x = self.relu(self.conv4(x))
    x = x.reshape(x.shape[0],-1)
    x = self.relu(self.lin1(x))
    x = self.relu(self.lin2(x))
    x = self.softmax(self.lin3(x))
    return x

