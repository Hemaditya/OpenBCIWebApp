# poo is responsible for retrieving data stored in Handmovment folder
import torch
from torch import optim
from lr_finder import LRFinder
import torch.nn as nn
from torch.optim import Adam
import eegnet
import poo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt


left_indices = np.arange(30,60)
right_indices = np.arange(90,120)
still_indices = np.append(np.arange(0,30),np.arange(60,90))


def classPolicy(theData, data_split=3, labels=[0,1,2],appendLabels=True):
	# How many splits does the data contain
	data = np.copy(theData)
	splitPoint = int(data.shape[0]/data_split)
	classLabels = np.zeros(shape=(data.shape[0]))
	for i, l in enumerate(labels):
		classLabels[i*splitPoint:(i+1)*splitPoint] = l
	
	if (appendLabels == True):
		classLabels = np.expand_dims(classLabels,0)
		classLabels = np.repeat(classLabels,data.shape[1],0)
		classLabels = np.transpose(classLabels,(1,0))

		data[...,-1] = classLabels
		return data
	
	else:
		return classLabels

	pass

class DS(torch.utils.data.Dataset):
	def __init__(self,data):
		self.data = torch.from_numpy(data).float()
	def __len__(self):
		return self.data.shape[0]
	def __getitem__(self,i):
		return (self.data[i,:,:,0:-1],self.data[i,0,0,-1].reshape(-1).squeeze().long())

def trainer(data):
	# Split it into test and train
	train, test = train_test_split(data,train_size=0.8)	
	train = train.reshape(train.shape[0],1,1,-1)	
	train = DS(train)
	dl = torch.utils.data.DataLoader(train,shuffle=True,batch_size=1)

	#train = np.expand_dims(train,1)
	test = np.expand_dims(test,1)
	test_labels, test_data = test[:,0,-1], test[:,:,0:-1]
	test_data = torch.from_numpy(test_data.reshape(test_data.shape[0],1,1,-1)).float()
	test_labels = test_labels.astype(int)
	epochs = 50
	net = eegnet.NN()
	optimizer = optim.Adam(net.parameters(), lr=1e-10, weight_decay=1e-2)
	criterion = nn.CrossEntropyLoss()
	lr_finder = LRFinder(net, optimizer, criterion)
	lr_finder.range_test(dl, end_lr=100, num_iter=100)
	lr_finder.plot()

	model.train()
	for i in range(epochs):
		lossItems = []
		np.random.shuffle(train)
		for j, sample in enumerate(train):
			label, sample = np.copy(sample[0,-1]),np.copy(sample[:,0:-1])
			label = torch.from_numpy(np.asarray(label)).long().unsqueeze(0)
			sample = torch.from_numpy(sample).float()
			optimizer.zero_grad()
			out = model(sample.reshape(1,1,1,-1))
			loss = criterion(out,label)
			loss.backward()
			optimizer.step()
			lossItems.append(loss.item())
			#print("Item no.: ",j+1,"/",train.shape[0]," Loss: ",loss.item())
		avgLoss = np.average(np.asarray(lossItems).reshape(-1))
		model.eval()
		out = torch.max(model(test_data),1)[1].reshape(-1)
		print(out,test_labels)
		a = accuracy_score(out, test_labels.reshape(-1))
		c = confusion_matrix(test_labels.reshape(-1),out)
		print("Accuracy: ",a)
		print(c)
		print("Epochs: ",i,", Loss: ",avgLoss)
	pass

data = np.copy(poo.daata)
data = data[2:]
#for i in range(data.shape[0]):
#	data[i] = normalize(data[i])
data = data[:,[6],:]
#data = data[0:100]
data = classPolicy(data)
#print(np.append(left_indices,right_indices).shape)
#data = data[np.append(left_indices,right_indices)]
#print(data[...,-1])
print(data.shape)
data =data.reshape(data.shape[0],-1)
fftData = np.abs(np.fft.rfft(data[:,0:-1],500))
fftData = normalize(fftData)
labels = data[:,-1].reshape(data.shape[0],-1)
fftData = np.hstack((fftData,labels))
#print(labels)
trainer(fftData)
