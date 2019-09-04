import torch.nn as nn

class NN(nn.Module):
  def __init__(self):
    super(NN,self).__init__()
    self.conv1 = nn.Conv1d(1,20,3,padding=1)
    self.conv2 = nn.Conv1d(20,40,3,padding=1)
    self.conv3 = nn.Conv1d(40,64,3,padding=1)
    self.conv4 = nn.Conv1d(64,32,3,padding=1)
    self.conv5 = nn.Conv1d(32,10,3,padding=1)
    self.lin1 = nn.Linear(100,2)
	self.d1 = nn.Drouput(0.5)
    self.relu = nn.ReLU()
    self.sig = nn.Sigmoid()
    self.softmax = nn.Softmax()
    
  def forward(self, x):
    x = self.sig(self.conv1(x))
    x = self.sig(self.conv2(x))
	x = self.d1(x)
    x = self.sig(self.conv3(x))
    x = self.sig(self.conv4(x))
	x = self.d1(x)
    x = self.sig(self.conv5(x)).reshape(x.shape[0],-1)
    x = self.softmax(self.lin1(x))
    
    return x
