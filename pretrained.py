import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models,datasets,transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
import os

model = models.resnet50()
model.load_state_dict(torch.load('resnet50-19c8e357.pth'))

def turn_off():
	for params in model.parameters():
		params.requires_grad = False
inp = model.fc.in_features
model.fc = nn.Linear(inp,3)

tf = transforms.Compose([
	transforms.ToPILImage(),
	transforms.Resize((224,224)),
	transforms.ToTensor()])

class SpectrogramDataset(Dataset):
	def __init__(self,data,transform=None):	
		
		# The Data is in the shape (batch_size,channels,fftOutputSize(127))
		self.data = data
		if(type(self.data) == np.ndarray):
			print("Converting numpy array to torch tensors......")
			self.data = torch.from_numpy(self.data).float()

		# Extract the labels from the data.
		# The output of the below operation will be (batch_size,channels)
		print("Extracting Labels......")
		self.labels = self.data[:,:,-1].long()
		self.data = self.data[:,:,0:-1]
		self.data = self.data/20000.0
		
		# Convert a single channel (1,8,126) data to (3,224,224)
		self.data = np.repeat(self.data,2,-1)
		print(self.data.shape)
		# replicate all 8 channels
		self.data = np.repeat(self.data,30,-2)
		print(self.data.shape)
		self.data = np.expand_dims(self.data,1)
		self.data = np.repeat(self.data,3,1)
		print(self.data.shape,type(self.data))
		self.data = torch.from_numpy(self.data)


		# Now just take the first value of the channels from all 8 channels 
		# because all the 8 channels have the same label
		self.labels
		ct = False
		for i, _ in enumerate(self.labels):
			if(all(self.labels[i,:] == self.labels[i,0])):
				ct = True
				pass
			else:
				ct = False
		if(ct == False):
			print("Unable to create Dataset because of incosistency of labels in channels")

		else:
			self.labels = self.labels[:,0]
			print("Labels extracted successfully.....")

	def __len__(self):
		return self.data.shape[0]
	
	def __getitem__(self,i):
		return (self.data[i],self.labels[i])	

def train(data,labels=[]):
	# You will have to seperate the dataset into train and test
	dataset = SpectrogramDataset(data)
	train_split = 0.8
	print("Splitting val and train sets: ",train_split,"....")
	train_size = int(train_split*len(dataset))
	val_size = len(dataset) - train_size

	print("Building train and val sets......")
	train_dataset, val_dataset = torch.utils.data.random_split(dataset,[train_size,val_size])
	print("Train and vest sets created successfully!")
	train_dataset, val_dataset = torch.utils.data.random_split(dataset,[train_size,val_size])

	print("Building train and val loaders.....")
	train_loader = DataLoader(train_dataset,batch_size=1, shuffle=True)
	val_loader = DataLoader(val_dataset,batch_size=4, shuffle=True)
	print("Train and val loaders built successfully")

	epochs = 10
	optimizer = optim.Adam(model.parameters(),lr=0.001)
	criterion = nn.NLLLoss()
	losses = []
	for e in range(epochs):
		model.train()
		for i,(d,l) in enumerate(train_loader):
			print("Training batch: ",i)
			print(d.shape)
			#d = tf(d.squeeze())
			optimizer.zero_grad()
			out = model(d)
			print("Calculating loss....")
			loss = criterion(out,l)	
			loss.backward()
			print("loss: ",loss.item())
			optimizer.step()
			losses.append(loss.item())
		print(sum(losses)/float(len(losses)))
		model.eval()
		out = model(val_loader.dataset.data).view(val_loader.dataset.data.shape[0],-1)
		pred = torch.max(out,1)[1]
		p = [pred == val_loader.dataset.labels.view(pred.shape[0])]
		acc = len(p == True)/float(len(p))
		print("VAL ACC: ",acc)
		
