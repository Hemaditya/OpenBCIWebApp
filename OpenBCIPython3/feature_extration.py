import mneProgram as MNE
import numpy as np
from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split

from keras import Sequential
from toCWT import toCWT
from keras.layers import Convolution2D,Dropout,BatchNormalization,MaxPool2D,AvgPool2D,Dense,Flatten
from keras.utils import np_utils

files = MNE.retrieveFiles()['pickle']
files.sort()
files = files[-1]
data, labels = MNE.dataFromFiles([files])
k = []
for l in labels:
	k.append(int(l)+1)
labels = np.asarray(k)

def buildLabels(labels):
	labels = labels.repeat(3).reshape(-1,3)
	labels = np.hstack((np.zeros(labels.shape,dtype=np.int),labels)).reshape(-1)
	return labels

def predict(data,labels):
	X,x,Y,y = train_test_split(data,labels,train_size=0.8,random_state=30)
	print("Train size: ",X.shape)
	print("Test size: ",x.shape)
	lda = LDA(n_components=8)
	X_ = lda.fit_transform(X.reshape(X.shape[0],-1),Y)
	x_ = lda.predict(x.reshape(x.shape[0],-1))

	print(x_)
	print(y)
	acc = np.mean((x_ == y))
	print("Accuracy: ",acc)

def buildModel(input_dim, out_labels):
	pass	

def waveletCNN(input_shape=(50,250,8),out_labels=2):
	'''
		Build a Image model to train Wavelete images
		- Will assume input shape of the image to be (50,250,8)
	'''
	model = Sequential()	

	model.add(Convolution2D(32,(3,3),padding='same',activation='relu',input_shape=input_shape)) #50,250,32
	model.add(BatchNormalization())
	model.add(Convolution2D(64,(3,3),padding='same',activation='relu')) #50,250,64
	model.add(BatchNormalization())
	model.add(MaxPool2D(2)) # 25,125,64
	model.add(Convolution2D(32,(3,3),padding='same',activation='relu')) # 25,125,32
	model.add(BatchNormalization())
	model.add(Convolution2D(64,(3,3),padding='same',activation='relu')) # 25,125,64
	model.add(BatchNormalization())
	model.add(MaxPool2D((2,2))) # 12,62,64
	model.add(Convolution2D(64,(3,3),padding='same',strides=2,activation='relu')) #6,31,64
	model.add(BatchNormalization())
	model.add(AvgPool2D((6,31)))
	model.add(Flatten())
	model.add(Dense(out_labels,activation='softmax'))	

	model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

	return model
	pass	

def trainWavelets(data,labels):
	'''
		YOu know what this is 
	'''

	data = toCWT(data)
	data = data.transpose(0,2,3,1)
	data = data[...,[2,3,4,5]]
	model = waveletCNN((50,250,4),2)
	X,x,Y,y = train_test_split(data,labels,train_size=0.8,random_state=40)
	Y = np_utils.to_categorical(Y)
	y = np_utils.to_categorical(y)

	model.fit(X,Y,epochs=50,shuffle=True,batch_size=4,validation_data=(x,y))


