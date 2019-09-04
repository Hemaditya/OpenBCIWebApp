import torch.nn as nn
import torch

class NN(nn.Module):
  def __init__(self):
    super(NN,self).__init__()
    
    self.conv1 = nn.Conv1d(250,32,1)
    self.conv2 = nn.Conv1d(32,64,1)
    self.conv3 = nn.Conv1d(64,128,1)
    self.max1 = nn.MaxPool1d(2)
    self.conv4 = nn.Conv1d(128,32,1)
    self.d = nn.Dropout(0.5)
    self.conv5 = nn.Conv1d(32,10,1)
    self.lin1 = nn.Linear(10,3)
    
    self.bn = nn.BatchNorm1d(32)
    self.bn2 = nn.BatchNorm1d(64)
    self.bn3 = nn.BatchNorm1d(10)
    self.relu = nn.ReLU()
    self.sig = nn.Sigmoid()
    self.softmax = nn.Softmax()
    
  def forward(self, x):
    x = self.sig(self.conv1(x))
    x = self.sig(self.conv2(x))
    x = self.sig(self.conv3(x))
    x = self.max1(x)
    x = self.sig(self.conv4(x))
    x = self.d(x)
    x = self.sig(self.conv5(x)).reshape(x.shape[0],-1)
    x = self.softmax(self.lin1(x))
    
    return x


