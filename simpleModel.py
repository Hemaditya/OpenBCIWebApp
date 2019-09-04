import torch.nn as nn


class NN(nn.Module):

	def __init__(self):
		super(NN,self).__init__()

		self.lin1 = nn.Linear(10,128)
		self.lin2 = nn.Linear(128,64)
		self.lin3 = nn.Linear(64,10)
		self.lin4 = nn.Linear(10,2)

		self.relu = nn.ReLU()
		self.softmax = nn.Softmax()

	
	def forward(self,x):
		x = x.reshape(-1)
		x = self.relu(self.lin1(x))
		x = self.relu(self.lin2(x))
		x = self.relu(self.lin3(x))
		x = self.softmax(self.lin4(x))

		return x
