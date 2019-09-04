import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class LSTMMod(nn.Module):

	def __init__(self,inp,hidd_size,out_size,class_size):
		super(LSTMMod,self).__init__()

		self.lstm = nn.LSTM(1,1,3)
		self.lin1 = nn.Linear(out_size,class_size)

		self.soft = nn.Softmax(dim=1)

	def forward(self,x):
		hstates = []
		h = torch.randn(3,1,1)
		c = torch.randn(3,1,1)
		h = (h,c)
		for i in x:
			i = torch.tensor(i)
			o,h = self.lstm(i.view(1,1,1),h)
			hstates.append(o)
		h = torch.cat(hstates)
		x = self.lin1(h.reshape(1,-1))
		x = self.soft(x)

		return x

class NormalModel(nn.Module):

	def __init__(self,inp):
		super(NormalModel,self).__init__()
		self.lin1 = nn.Linear(inp,160)
		self.lin2 = nn.Linear(160,100)
		self.lin3 = nn.Linear(100,3)

		self.sig = nn.Sigmoid()
		self.soft = nn.Softmax(dim=1)
	
	def forward(self,x):
		x = x.reshape(1,-1)
		x = self.lin1(x)
		x = self.sig(x)
		x = self.lin2(x)
		x = self.sig(x)
		x = self.lin3(x)
		x = self.soft(x)

		return x

def modelTrain(d):
	train,test = train_test_split(d,train_size=0.8)	

	train = torch.from_numpy(train)
	labels,test = test[:,-1],test[:,0:-1]
	test = torch.from_numpy(test).float()

	epochs = 50
	lr = 3e-4
	model = LSTMMod(1,1,11,3)
	optim = torch.optim.Adam(model.parameters(),lr=lr)
	criterion = nn.CrossEntropyLoss()

	for i in range(epochs):
		lossItems = []
		model.train()
		for j, a in enumerate(train):
			optim.zero_grad()
			label,a = torch.tensor([a[-1]]),a[0:-1]
			out = model(a.float())
			loss = criterion(out,label.long())
			loss.backward()
			optim.step()
			lossItems.append(loss.item())
		model.eval()		
		outs = []
		lo = np.average(np.asarray(lossItems))
		for j in test:
			outs.append(torch.max(model(j),1)[1])
		acc = accuracy_score(np.asarray(outs),labels.reshape(-1))			
		print("Epoch: ",i+1,", Accuracy Score: ",acc,", Loss: ",lo)

def test():
	data = np.random.randint(0,3,size=(100,12))
	modelTrain(data)

