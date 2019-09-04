import torch.nn as nn
import torch
from torch.optim import Adam

class NN(nn.Module):

	def __init__(self):
		
		super(NN,self).__init__()
		self.lin1 = nn.Linear(8,16)
		self.lin2 = nn.Linear(16,10)
		self.lin3 =	nn.Linear(10,3)

		self.sig = nn.Sigmoid()
	
	def forward(self,x):
		x = self.sig(self.lin1(x))	
		x = self.sig(self.lin2(x))	
		x = self.sig(self.lin3(x))	

		return x


