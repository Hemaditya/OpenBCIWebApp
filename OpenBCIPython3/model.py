import keras
from keras import Sequential
from keras.layers import Convolution2D as Conv2d
from keras.layers import Dense,Dropout,MaxPooling2D,AvgPool2D,Flatten
inp_shape = (8,40,1)
model = Sequential()
model.add(Conv2d(32,(1,3),padding='same',activation='relu',input_shape=inp_shape)) # 8,60,16
model.add(Conv2d(32,(1,3),padding='same',activation='relu')) # 8,60,32
model.add(Dropout(0.3))
model.add(MaxPooling2D((2,2))) # 4,20,32
model.add(Conv2d(64,(1,3),padding='same',activation='relu')) # 4,20,16
model.add(Conv2d(64,(1,3),padding='same',activation='relu')) # 4,20,32
model.add(MaxPooling2D((2,2))) # 2,10,32
model.add(Conv2d(64,(1,3),strides=(1,2),activation='relu')) # 2,5,16
model.add(AvgPool2D((2,4)))
model.add(Flatten())
model.add(Dense(2,activation='softmax'))
print(model.summary())

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
