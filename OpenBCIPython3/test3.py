import mneProgram as MNE
from sklearn.preprocessing import StandardScaler,normalize
import numpy as np
from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import time
import keras 
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils

files = MNE.retrieveFiles()
files = files["pickle"]
acc1 = 0
band =  [0,0]
a1 = range(30,12,-2)
a2 = range(0,1,1)
model = []
mean = []
std = []
for i in a2:
	data,labels = MNE.dataFromFiles(files)
	uVolts_per_count = (4.5)/24/(2**23-1)*1000000 # scalar factor to convert raw data into real world signal data
	data = data*uVolts_per_count
	#for i, sample in enumerate(data):
	#	data[i] = normalize(sample)
	data = MNE.applyFilters(data)
	#data = data[:,[3,4],:]
	indices = list(range(3,data.shape[0],6))
	indices = np.asarray([[i,i+1,i+2] for i in indices]).reshape(-1)
	#indices = np.asarray([[i+1,i+2] for i in indices]).reshape(-1)
	#indices = np.asarray([[i] for i in indices]).reshape(-1)
	#data = data[indices]
	labels = np.repeat(labels,3).reshape(-1)
	labels = labels.reshape(-1)
	print(labels.shape)
	for i, j in enumerate(labels):
		labels[i] = int(j)+1
	labels = labels.reshape(-1,3)
	zeros = np.zeros(shape=labels.shape,dtype=np.int)
	labels = np.hstack((zeros,labels))
	labels = labels.reshape(-1)
	#@data1 = data[labels == '1']
	#@data2 = data[labels == '2']
	#@data = np.vstack((data1,data2))
	#@labels1  = labels[labels == '1']
	#@labels2  = labels[labels == '2']
	#@labels = np.append(labels1,labels2)
	labels[labels == '1'] = 1
	labels[labels == '2'] = 2
	labels[labels == '3'] = 1
	labels[labels == '4'] = 2

	classBalance = np.mean(labels == labels[0])
	print("CLASS BALANCE: ",max(classBalance, 1.0 - classBalance))

	#csp = CSP(n_components=8)
	#data = csp.fit_transform(data,labels)		
	mean = np.mean(data)
	std = np.std(data)
	data = (data - np.mean(data).reshape(-1))/np.std(data)
	labels = labels.astype('int')
	l = []
	for la in labels:
		l.append(int(la))
	labels = np.array(l)
	labels = np_utils.to_categorical(labels)
	print(labels.shape)
	
	model = Sequential()
	data = data.reshape(data.shape[0],-1)
	print(data.shape)
	print(labels.shape)
	print(np.unique(labels))
	print(data.shape[-1])
	X,x,Y,y = train_test_split(data,labels,shuffle=True,train_size=0.70)
	model.add(Dense(1000,activation='relu',input_dim=data.shape[-1]))
	model.add(Dropout(0.35))
	model.add(Dense(160,activation='relu'))
	model.add(Dropout(0.35))
	model.add(Dense(160,activation='relu'))
	model.add(Dropout(0.35))
	model.add(Dense(3,activation='softmax'))
	model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
	model.fit(X,Y,epochs=20,batch_size=8,validation_data=(x,y),shuffle=True)
