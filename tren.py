from sklearn.model_selection import train_test_split
from torch.optim import Adam
import torch.nn as nn
import torch
import numpy as np
import model
import poo
import time
from sklearn.metrics import accuracy_score

deita = np.copy(poo.daata)

def train(data):
	m = model.NN()
	train, test = train_test_split(data,train_size=0.8)

	#x_train, y_train = train[:,:,0:-1],train[:,0,-1]
	#x_test, y_test = test[:,:,0:-1],test[:,0,-1]
	
	
	epochs = 40
	optimizer = Adam(m.parameters(),lr=3e-4)
	criterion = nn.CrossEntropyLoss()
	test = torch.from_numpy(test)
	labels = test[:,0,-1]
	b_test = test[:,:,0:-1]
	b_test[b_test>=500.0] = 500.0
	b_test[b_test<=-500.0] = -500.0
	b_test = b_test/500.0
	
	for i in range(epochs):
		np.random.shuffle(train)
		m.train()
		losses = []
		for j, d in enumerate(train):
			l = d[...,0,-1].reshape(-1)
			#print(l)
			d[d>=500.0] = 500.0
			d[d<=-500.0] = -500.0
			d = d/500.0
			d = torch.from_numpy(d).float().unsqueeze(0)
			l = torch.from_numpy(l).long()
			optimizer.zero_grad()
			out = m(d)
			loss = criterion(out,l)
			losses.append(loss.item())
			loss.backward()
			optimizer.step()
		losses = np.array(losses)
		print(np.average(losses))
		m.eval()
		out = m(b_test.float())
		print(accuracy_score(labels, torch.max(out,1)[1]))

train(deita)
