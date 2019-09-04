import torch.nn as nn


class NN(nn.Module):

	def __init__(self):
		super(NN,self).__init__()

		self.conv1 = nn.Conv2d(1,10,(1,3),padding=(0,1)) # 500, 2
		self.conv2 = nn.Conv2d(10,20,(1,3),padding=(0,1)) # 500, 4
		self.bn1 = nn.BatchNorm2d(20)
		self.mp1 = nn.MaxPool2d((1,2)) # 250, 8
		self.conv3 = nn.Conv2d(20,20,(1,3),padding=(0,1)) # 250,10
		self.conv4 = nn.Conv2d(20,30,(1,3),padding=(0,1)) # 250,12
		self.bn2 = nn.BatchNorm2d(30)
		self.mp2 = nn.MaxPool2d((1,2)) # 125, 24
		self.conv5 = nn.Conv2d(30,30,(1,3),padding=(0,1)) # 
		self.conv6 = nn.Conv2d(30,40,(1,3)) # 118,28
		self.bn3 = nn.BatchNorm2d(40)
		self.mp3 = nn.MaxPool2d((1,2)) # 59, 56
		self.conv5 = nn.Conv2d(40,30,(1,3)) # 57, 58
		self.conv6 = nn.Conv2d(30,25,(1,3)) # 55, 60
		self.mp4 = nn.MaxPool2d((1,2)) # 27, 120
		#self.conv7 = nn.Conv2d(
		self.lin1 = nn.Linear(251,100)
		self.lin2 = nn.Linear(100,30)
		self.lin3 = nn.Linear(30,3)

		self.act = nn.ReLU()
		self.soft = nn.Softmax()
	
	def forward(self,x):
		
		#x = self.act(self.conv1(x))
		#x = self.act(self.conv2(x))
		#x = self.bn1(x)
		#x = self.mp1(x)
		#x = self.act(self.conv3(x))
		#x = self.act(self.conv4(x))
		#x = self.bn2(x)
		#x = self.mp2(x)
		#x = self.act(self.conv5(x))
		#x = self.act(self.conv6(x))
		#x = self.bn3(x)
		#x = self.mp3(x)

		x = x.reshape(x.shape[0],-1)
		x = self.act(self.lin1(x))
		x = self.act(self.lin2(x))
		x = self.soft(self.lin3(x))

		return x
