import torch.nn as nn


class NN(nn.Module):
	
	def __init__(self):
		super(NN,self).__init__()

		# size = (,1,126)
		self.conv1 = nn.Conv2d(8,32,(1,3))
		# size = (10,1,124)
		self.conv2 = nn.Conv2d(32,64,(1,3))
		# size = (20,1,122)
		self.m1 = nn.MaxPool2d((1,2))
		# size = (20,1,61)
		self.conv3 = nn.Conv2d(64,32,(1,3))
		# size = (20,1,59)
		self.conv4 = nn.Conv2d(32,32,(1,3))
		# size = (30,1,57)
		self.m2 = nn.MaxPool2d((1,2))
		# size = (30,1,28)
		self.conv5 = nn.Conv2d(32,10,(1,3))
		# size = (10,1,26)
		self.lin1 = nn.Linear(10*26,64)
		self.lin2 = nn.Linear(64,3)

		self.relu = nn.ReLU()
		self.softmax = nn.Softmax()
	
	def forward(self, x):
		x = self.relu(self.conv1(x))
		x = self.relu(self.conv2(x))
		x = self.m1(x)
		x = self.relu(self.conv3(x))
		x = self.relu(self.conv4(x))
		x = self.m1(x)
		x = self.relu(self.conv5(x)).reshape(x.shape[0],-1)
		x = self.relu(self.lin1(x))
		x = self.softmax(self.lin2(x))

		return x
