import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from fastai import *
from fastai.vision import *
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch
data = np.load("Stuff.npy")
from torchvision import models
resnet18 = model.resnet18(pretrained=True)

class DS(Dataset):
	
	def __init__(self,data,labels):
		
		self.data = torch.from_numpy(data)
		self.labels = torch.from_numpy(labels)
	
	def __len__(self):
		return self.labels.shape[0]
	
	def __getitem__(self,idx):
		return self.data[idx],self.labels[idx]

	
zeros = np.zeros(shape=(int(data.shape[0]/2),1)).reshape(-1,3)
ones = np.ones(shape=(int(data.shape[0]/2),1)).reshape(-1,3)
labels = np.hstack((zeros,ones)).reshape(-1)

print(labels.shape[0] == data.shape[0])
print(labels.shape, data.shape)

np.random.seed(42)
X,x,Y,y = train_test_split(data,labels,train_size=0.8)
train_ds = DS(X,Y)
valid_ds = DS(x,y)

train_dl = DataLoader(train_ds,shuffle=True,batch_size=1)
valid_dl = DataLoader(valid_ds,shuffle=True,batch_size=1)

last_layer = nn.Linear(512,2)
def freeze(model):
	for param in model.parameters():
		param.requires_grad = False
	return model

def trainModel(train_dl, valid_dl):
	
	epochs = 20
	lr = 0.003
	model = resnet18
	model = freeze(model)
	model.fc = last_layer	
	optimizer = torch.optim.Adam(model.parameters(),lr=lr)
	criterion = torch.nn.CrossEntropyLoss()
	for i in range(epochs):
	
