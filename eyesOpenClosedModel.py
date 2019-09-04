import torch.nn as nn

class M(nn.Module):
	def __init__(self):
		super(M,self).__init__()

		self.l1 = nn.Conv1d(20,32,1)
		self.l2 = nn.Conv1d(32,10,1)
		self.l3 = nn.Linear(10,2)

		self.sig = nn.Sigmoid()
		self.relu = nn.ReLU()
		self.soft = nn.Softmax()

	def forward(self,x):

		x = self.sig(self.l1(x))
		x = self.sig(self.l2(x)).reshape(x.shape[0],-1)
		x = self.sig(self.l3(x))
		x = self.soft(x)
		return x	


