import torch.nn as nn


class NN(nn.Module):
	def __init__(self):
		super(NN,self).__init__()
		self.lin1 = nn.Linear(64,10)
		self.lin2 = nn.Linear(10, 2)
		self.d1 = nn.Dropout()
		self.relu = nn.ReLU()
		self.sig = nn.Sigmoid()
		self.softmax = nn.Softmax()

	def ConvBlock(self,x,inp,out,filt):
		conv = nn.Conv1d(inp,out,filt)
		x = self.relu(conv(x))
		return x	


	def ResnetBlock(self,x,inp,out,fil):
		x = self.ConvBlock(x,inp,out,1)
		temp = x
		x = self.ConvBlock(x,out,out,1)
		x = self.ConvBlock(x,out,out,1)
		return x+temp
    
	def forward(self, x):
		x = self.ResnetBlock(x,10,32,1)
		x = self.ResnetBlock(x,32,64,1)
		x = x.reshape(x.shape[0],-1).double()
		x = self.relu(self.lin1(x))
		x = self.softmax(self.lin2(x))
		return x
