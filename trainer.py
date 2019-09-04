import torch
from torch.utils.data import DataLoader as DL
import torch.nn as nn
from torch.optim import Adam
from torch.optim import SGD
from torchvision import transforms
import matplotlib.pyplot as plt
import time
import trainModel
import os
import handMovemenModelCNN as w
model_name = 'EOEC'
model = w.NN()
model = model.double()

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.1 ** (epoch // 30)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(data,inp_size=126):
	#if(data.reshape(-1).shape[0] % 250 != 0):
	#	print("Data set not compatible for training")
	#	os._exit(0)
	#else:
	optimizer = Adam(model.parameters(),lr=1e-4)
	criterion = nn.CrossEntropyLoss()
	epochs = 40
	data = torch.from_numpy(data).float()
	print(data.shape)
	#trans = transforms.Compose([
	#	transforms.ToPILImage(),
	#	transforms.Resize(
	t = transforms.ToPILImage()
	tt = transforms.Resize(224)
	ttt = transforms.ToTensor()
	y = time.time()
	newT = tt(t(data[2]))
	print(y-time.time())
	print(newT)

	plt.imshow(t(data[2]))
	plt.show()
	rawNew = DL(data[0:data.shape[0]/3],shuffle=True,batch_size=1)	
	raw = DL(data[data.shape[0]/3:],shuffle=True,batch_size=1)	

	for e in range(epochs):
		losses = []
		model.train()

		for sample in raw:
			#sample = sample.reshape(-1)
			c = sample[0,:,-1][0].long().reshape(-1)
			sample = sample[0,:,0:-1]/20000.0
			sample = sample.unsqueeze(0)	
			optimizer.zero_grad()
			out = model(sample.unsqueeze(2))
			out = out.reshape(1,-1)
			loss = criterion(out,torch.tensor([c]))
			loss.backward()
			optimizer.step()
			losses.append(loss.item())
		total = sum(losses)/len(losses)
		print("Epoch: ",e+1,", Total loss: ",total)
		accuracy = []
		model.eval()
		for i in rawNew:
			model_val = model(((i[0,:,0:-1]/20000.0).unsqueeze(0)).unsqueeze(2)).argmax().item() == i[0,:,-1][0]
			accuracy.append(model_val)
		acc = accuracy.count(True)/float(len(accuracy))
		print("accuracy: ",acc)
	torch.save(model,model_name)
	torch.save(model,model_name+"backup")
	return model
