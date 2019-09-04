import keras
from keras import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import numpy as np

def denseModel(inp):
	# Build the model
	model = Sequential()
	model.add(Dense(1024,activation="sigmoid",input_dim=inp))
	model.add(Dense(1024,activation="sigmoid"))
	model.add(Dense(1024,activation="sigmoid"))
	model.add(Dense(3,activation='softmax'))

	model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
	return model

def modelTrain(data):
	trainData, labels = data[:,:,0:-1],data[0,:,-1]
	# Change the shape of data from (channels,samples,chunk_size) to (samples,channels,chunk_size) 
	trainData = np.transpose(trainData,(1,0,2))
	# Change the shape of data from (samples,channels,chunk_size) to (samples,channels*chunk_size)
	trainData = trainData.reshape(trainData.shape[0],-1)
	print(trainData.shape,labels.shape)
	
	# Build the model
	model = denseModel(250*3) 

	labels = np_utils.to_categorical(labels,3)

	model.fit(trainData,labels,batch_size=1,epochs=50,shuffle=True,validation_split=0.3)
