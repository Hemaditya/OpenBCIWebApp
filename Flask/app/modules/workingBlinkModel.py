import torch.nn as nn

class NN(nn.Module):
  def __init__(self):
    super(NN,self).__init__()
    self.conv1 = nn.Conv1d(10,32,1)
    self.conv2 = nn.Conv1d(32,32,1)
    self.conv3 = nn.Conv1d(32,32,1)
    self.conv4 = nn.Conv1d(32,10,1)
    self.conv5 = nn.Conv1d(64,32,1)
    self.conv6 = nn.Conv1d(32,10,1)
    self.lin1 = nn.Linear(10,2)
    self.d1 = nn.Dropout()
    self.relu = nn.ReLU()
    self.sig = nn.Sigmoid()
    self.softmax = nn.Softmax()
    
  def forward(self, x):
    x = self.sig(self.conv1(x))
    x1 = x
    x = self.sig(self.conv2(x)) + x1
    x2 = x
    x = self.sig(self.conv3(x))+x2
    x = self.sig(self.conv4(x)).reshape(x.shape[0],-1)
#    x = self.sig(self.conv5(x)) + x1
#    x = self.sig(self.conv6(x)).reshape(-1,10)
    x = self.softmax(self.lin1(x))
    
    return x
