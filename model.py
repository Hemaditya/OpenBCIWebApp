import torch
import torch.nn as nn

class NN(nn.Module):
	
	def __init__(self):
		super(NN,self).__init__()

		#250
		self.c1 = nn.Conv1d(8,16,11)
		#240
		self.c2 = nn.Conv1d(16,32,11)
		#230
		self.m1 = nn.MaxPool1d((2))
		#115
		self.c3 = nn.Conv1d(32,64,11)
		#105
		self.c4 = nn.Conv1d(64,86,11)
		#95
		self.m2 = nn.MaxPool1d((2))
		#47
		self.c5 = nn.Conv1d(86,128,11)
		#37
		self.c6 = nn.Conv1d(128,160,11)
		#27
		self.l1 = nn.Linear(160*27,200)
		self.l2 = nn.Linear(200,30)
		self.l3 = nn.Linear(30,3)

		self.sig = nn.Sigmoid()
		self.soft = nn.Softmax()
	
	def forward(self,x):

		x = self.sig(self.c1(x))
		x = self.sig(self.c2(x))
		x = self.m1(x)
		x = self.sig(self.c3(x))
		x = self.sig(self.c4(x))
		x = self.m2(x)
		x = self.sig(self.c5(x))
		x = self.sig(self.c6(x))
		x = x.reshape(x.shape[0],-1)
		x = self.sig(self.l1(x))
		x = self.sig(self.l2(x))
		x = self.soft(self.l3(x))

		return x
