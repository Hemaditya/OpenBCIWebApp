import poo
import scipy.signal as signal
import torch
from torch.optim import Adam
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
ct = 0
import matplotlib.pyplot as plt
from matplotlib.mlab import specgram
import numpy as np
#from handMovemenModel import NN

deita = np.copy(poo.daata)
deita = deita[2:]

print(deita.shape)
#left_channel_data = np.copy(deita[range(4,60,5)])
#right_channel_data = np.copy(deita[range(64,120,5)])
left_channel_data = np.copy(deita[:60,:,0:-1])
right_channel_data = np.copy(deita[60:,:,0:-1])


left_channel_data = left_channel_data[1:]
right_channel_data = right_channel_data[1:]

left_hand_data_only = np.copy(left_channel_data[[i for i in range(4,left_channel_data.shape[0],5)]])
right_hand_data_only = np.copy(right_channel_data[[i for i in range(4,right_channel_data.shape[0],5)]])
no_data_only = np.copy(deita[[i for i in range(120) if (i+1)%5 != 0]])

#left_channel_data = np.copy(deita[:60])
#right_channel_data = np.copy(deita[60:])
print(left_channel_data.shape)
print(right_channel_data.shape)

left_channel_data = np.transpose(left_channel_data,(1,0,2))
right_channel_data = np.transpose(right_channel_data,(1,0,2))

left_hand_data_only = np.transpose(left_hand_data_only,(1,0,2))
right_hand_data_only = np.transpose(right_hand_data_only,(1,0,2))
no_data_only = np.transpose(no_data_only,(1,0,2))

print(left_channel_data.shape)
print(right_channel_data.shape)

#left_channel_data = left_channel_data.reshape(left_channel_data.shape[0],-1)
#right_channel_data = right_channel_data.reshape(right_channel_data.shape[0],-1)

#left_channel_data = left_channel_data.reshape(left_channel_data.shape[0],12,-1)
#right_channel_data = right_channel_data.reshape(right_channel_data.shape[0],12,-1)
#
#left_channel_data = left_channel_data[:,1:,:]
#right_channel_data = right_channel_data[:,1:,:]

#print(left_channel_data)

print(left_channel_data.shape)
print(right_channel_data.shape)

def buildDatasetForSingleChannel(ch_lh, ch_rh):
	print(ch_lh.shape)
	print(ch_lh.reshape(-1).shape)
	data1 = ch_lh.reshape(-1,251)
	data2 = ch_rh.reshape(-1,251)
	#labels_left = np.repeat([[0,0,0,0,1]],11,0).reshape(-1,1)
	#labels_right = np.repeat([[0,0,0,0,2]],11,0).reshape(-1,1)

	data1 = np.hstack((data1,labels_left))
	data2 = np.hstack((data2,labels_right))
	data = np.vstack((data1,data2))
	empty = np.empty(shape=(1,127))
	print(empty.shape)
	for d in data:
		x,y,z = specgram(d[0:-1],250,250)
		x = np.append(x,d[-1]).reshape(1,-1)
		empty = np.vstack((empty,x))
	empty = empty[1:]
	return empty
	
Zstate = {}
Zstate['notch'] = {}
Zstate['dc_offset'] = {}
Zstate['bandpass'] = {}
for c in range(8):
	Zstate['notch'][c] = [0,0,0,0,0,0]
	Zstate['dc_offset'][c] = [0,0]
	Zstate['bandpass'][c] = [0,0,0,0,0,0]

#def bandpass(arr,state):
#		# This is to allow the band of signal to pass with start frequency and stop frequency
#		start = 14
#		stop = 30
#		bp_Hz = np.zeros(0)
#		bp_Hz = np.array([start,stop])
#		b, a = signal.butter(3, bp_Hz/(250 / 2.0),'bandpass')
#		bandpassOutput, state = signal.lfilter(b, a, arr, zi=state)
#		return bandpassOutput, state
#
#for c in range(8):
#	for i in range(left_channel_data.shape[1]):
#		bdp = left_channel_data[c,i].reshape(-1)
#		l = bdp[-1]
#		bdp = bdp[0:-1]
#		bandpassOutput, Zstate['bandpass'][c] = bandpass(bdp,Zstate['bandpass'][c])
#		left_channel_data[c,i] = np.append(bandpassOutput,l)
#
#for c in range(8):
#	for i in range(right_channel_data.shape[1]):
#		bdp = right_channel_data[c,i].reshape(-1)
#		l = bdp[-1]
#		bdp = bdp[0:-1]
#		bandpassOutput, Zstate['bandpass'][c] = bandpass(bdp,Zstate['bandpass'][c])
#		right_channel_data[c,i] = np.append(bandpassOutput,l)

def trainModel(data):
	train, test = train_test_split(data,train_size=0.7)	
	model = NN()
	epochs = 50
	optimizer = Adam(model.parameters(),lr = 4e-6)
	criterion = nn.CrossEntropyLoss()	
	test = torch.from_numpy(test).float()
	for i in range(epochs):
		lossItems = []
		np.random.shuffle(train)
		model.train()
		for sample in train:
			sample = torch.from_numpy(sample)
			label = torch.tensor(sample[-1])
			label = label.reshape(1).long()
			sample = sample[0:-1].float().reshape(1,-1)
			optimizer.zero_grad()
			out = model(sample)
			loss = criterion(out,label)
			loss.backward()
			optimizer.step()
			lossItems.append(loss.item())
		avg = np.average(np.asarray(lossItems))
		print("Epoch: ",i,", Loss: ",avg)
		model.eval()
		out = model(test[:,0:-1])
		labels = torch.max(out,1)[1].reshape(-1).numpy()
		print(confusion_matrix(labels.reshape(-1),test[:,-1].reshape(-1)))
#print("Building Dataset")
#datasetData = buildDatasetForSingleChannel(left_channel_data[0],right_channel_data[0])
#print("Sending to the train")
#trainModel(datasetData)
	
def newFunc():
	fig,ax = plt.subplots(2,2)
	ax = ax.reshape(-1)
	print(ax.shape)
	#plt.subplots_adjust(wspace=0.5,hspace=1.0)
	#for i in [2]:
	#	for j, d in enumerate(left_channel_data[i]):
	#		ax.plot(d.reshape(-1),alpha=0.5)
	#Qplt.plot(left_channel_data[0,1:,:].reshape(-1))
	#plt.show()
	from matplotlib.mlab import specgram

	for i,j in enumerate([3,4,6]):
		c1 = left_channel_data[j,:,:]
		c2 = right_channel_data[j,:,:]
		c3 = no_data_only[j,:,:]
		#x,y,z = specgram(c3.reshape(-1),NFFT=250,Fs=250,detrend='mean')
		x = np.rot90(np.abs(np.fft.rfft(c1)))
		x1,y1,z1 = specgram(c1.reshape(-1),NFFT=250,Fs=250)
		x = x*250.0/float(250)
		x1 = x1*250.0/float(250)
		#print(x.shape,y.shape,z.shape)
		#ax[0].cla()
		#ax[1].cla()
		#ax[0].pcolormesh(z,y,10*np.log10(x),vmin=-26,vmax=25)
		#ax[0].title.set_text("left hand")
		#ax[1].pcolormesh(z1,y1,10*np.log10(x1),vmin=-26,vmax=25)
		#ax[1].title.set_text("right hand")
		#ax[i].pcolormesh(z,y,10*np.log10(x),vmin=-26,vmax=25)
		ax[i].imshow(10*np.log10(x),vmin=-25,vmax=26)
		ax[i].title.set_text("Channel_"+str(j))

	fig.suptitle("no_data")
	plt.savefig("no_data")
		#plt.pcolormesh(z,y,10*np.log10(x),vmin=-26,vmax=25)
	plt.show()
newFunc()


def energy():
	left_log_energy = {}
	right_log_energy = {}
	#fig,ax = plt.subplots(4,2)
	#ax = ax.reshape(-1)
	#print(ax.shape)
	for c in range(left_channel_data.shape[0]):
		allData = left_channel_data[c]
		allData = np.copy(left_channel_data[c].reshape(-1,10))
		left_log_energy[c] = []
		for data in allData:
			sqr = np.square(data.reshape(-1))
			Sum = np.sum(sqr)
			log = 10*np.log(Sum)
			left_log_energy[c].append(log)

	for c in range(right_channel_data.shape[0]):
		allData = right_channel_data[c]
		right_log_energy[c] = []
		allData = np.copy(right_channel_data[c].reshape(-1,10))
		for data in allData:
			sqr = np.square(data.reshape(-1))
			Sum = np.sum(sqr)
			log = 10*np.log(Sum)
			right_log_energy[c].append(log)

	#print(left_log_energy)
	#print(right_log_energy)
	#plt.plot(left_log_energy[0])
	#plt.plot(right_log_energy[0])

	d = np.zeros(shape=len(left_log_energy[0]))
	for i in range(8):
		if(i+1 != 6 and i+1 != 8):
			print(len(left_log_energy[i]))
			print(len(right_log_energy[i]))
			d = d + np.asarray(left_log_energy[i])
		#ax[i].hist(left_log_energy[i],alpha=0.7)
		#ax[i].hist(right_log_energy[i],alpha=0.7)
		#ax[i].plot(left_log_energy[i][1:],alpha=0.7)
		#ax[i].plot(right_log_energy[i][1:],alpha=0.7)
		#ax[i].title.set_text(str(i+1))
	plt.plot(right_channel_data[0].reshape(-1))
	#plt.plot(right_log_energy[0][1:],alpha=0.7,c='b')
	plt.show()
#energy()
