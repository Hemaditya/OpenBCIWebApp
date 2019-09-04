import torch
from torch.utils.data import DataLoader as DL
import torch.nn as nn
from torch.optim import Adam
import trainModel
import os
model_name = 'Artifact Removal'
model = torch.load(model_name)
model = model.double()

def train(data,inp_size=250):
	#if(data.reshape(-1).shape[0] % 250 != 0):
	#	print("Data set not compatible for training")
	#	os._exit(0)
	#else:
	optimizer = Adam(model.parameters(),lr=0.001)
	criterion = nn.CrossEntropyLoss()
	epochs = 50
	data = torch.from_numpy(data)
	raw = DL(data,shuffle=True,batch_size=1)	
	model.train()

	for e in range(epochs):
		losses = []
		for sample in raw:
			sample = sample.reshape(-1)
			c = sample[-1].long()
			sample = sample[0:-1]
			sample = sample.reshape(1,inp_size,1)
			optimizer.zero_grad()
			out = model(sample)
			loss = criterion(out,torch.tensor([c]))
			loss.backward()
			optimizer.step()
			losses.append(loss.item())
		total = sum(losses)/len(losses)
		print("Total loss: ",total)
	torch.save(model,model_name)
	torch.save(model,model_name+"backup")
	return model
