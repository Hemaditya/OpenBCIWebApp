import os
import torch
import pickle
import numpy as np
from torchvision import transforms
import time
#import matplotlib.pyplot as plt
data = []
#plt.legend()
#fig,ax = plt.subplots(2,4)
#ax = ax.reshape(-1)
chunk_size= 250

f = os.listdir('HandMovement/')

xt = 0
if xt == 1:
	for fi in f:
		with open('HandMovement/'+fi,'rb') as g:
			data.append(pickle.load(g,encoding='latin1'))

elif xt == 0:
	for fi in f:
		with open('HandMovement/'+fi,'r') as g:
			data.append(pickle.load(g))
	

for i,d in enumerate(data):
    for k in d:
        for j,l in enumerate(d[k]):
            data[i][k][l][:,-1] = k
            

channels = {}
for i in range(8):
	channels[i] = np.zeros(shape=(1,chunk_size+1))

for i,d in enumerate(data):
	for k in d:
		for c in d[k]:
			channels[c] = np.vstack((channels[c],d[k][c]))
			pass

deita = channels[1]
daata = np.zeros(shape=(1,8,chunk_size+1))
for i in range(deita.shape[0]):
	newD = channels[0][i]
	for j in channels:
		if j>0:
			newD = np.vstack((newD,channels[j][i]))
	newD = np.expand_dims(newD,axis=0)
	daata = np.vstack((daata,newD))

#daata[daata>=400.0] = 400.0
#daata[daata<=-400.0] = -400.0
#daata = daata + 500.0
#f = np.fft.rfftfreq(250)*250.0
#do = np.zeros(shape=(50,60))
#for d in daata:
#	print(d[0,-1])
#	do = np.roll(do,-8,0)
#	d = d/20000.0
#	do[-8:] = d[:,0:60]
#	#plt.clf()
#	#plt.imshow(do).set_clim(0,1.0)
#	#plt.draw()
#	#x= raw_input("S: ")	
	
#toImg = transforms.ToPILImage()
#toRes = transforms.Resize((500,500))
#daata = daata[1:]
#yticks = np.arange(45,0,-1)
#time = 200
#strip_times = 3
#channels_times = 3
#pl = np.zeros(shape=(time*strip_times,8,45*channels_times))

#print("Importing Fastai....")
#import pretrained as f
#print("Sending for training process.....")
#f.train(daata)
