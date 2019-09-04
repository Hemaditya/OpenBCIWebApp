import poo
import keras
from keras import Sequential
from keras.layers import Dense,Dropout
from sklearn.preprocessing import normalize
import numpy as np

d = np.copy(poo.daata)

d = d[2:]

d = d[:,[3,4,6],0:-1]

#d[:,[0,2],0:-1]  = np.expand_dims(d[:,1,0:-1],1)- d[:,[0,2],0:-1]

fftD = np.abs(np.fft.rfft(d))


fftD = fftD[:,2,:40]
print(fftD.shape)

model = Sequential()
model.add(Dense(50,activation='sigmoid',input_dim=40))
model.add(Dense(50,activation='sigmoid'))
model.add(Dense(50,activation='sigmoid'))
model.add(Dense(3,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

X_train = fftD[:,:].reshape(fftD.shape[0],-1)
X_train
X_train = normalize(X_train)
Y_train = np.array([[0]*15,[1]*15,[2]*15]*4).reshape(-1)

from keras.utils import np_utils
Y_train = np_utils.to_categorical(Y_train,3)

#model.fit(X_train,Y_train,batch_size=1,epochs=50,shuffle=True,validation_split=0.3)
model.save("Aditya Model.h5")
