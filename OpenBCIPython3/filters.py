from scipy import signal
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.tensor

from keras import Sequential 
from keras.layers import Dense
from mne.decoding import CSP
def bandpass(arr,state):
		# This is to allow the band of signal to pass with start frequency and stop frequency
		start = 1
		stop = 14
		bp_Hz = np.zeros(0)
		bp_Hz = np.array([start,stop])
		b, a = signal.butter(3, bp_Hz/(250 / 2.0),'bandpass')
		bandpassOutput, state = signal.lfilter(b, a, arr, zi=state)
		return bandpassOutput, state

def notchFilter(arr,state):
	# This is to remove the AC mains noise interference	of frequency of 50Hz(India)
	notch_freq_Hz = np.array([50.0])  # main + harmonic frequencies
	for freq_Hz in np.nditer(notch_freq_Hz):  # loop over each target freq
		bp_stop_Hz = freq_Hz + 3.0*np.array([-1, 1])  # set the stop band
		b, a = signal.butter(3, bp_stop_Hz/(250 / 2.0), 'bandstop')
		notchOutput, state = signal.lfilter(b, a, arr, zi=state)
		return notchOutput, state

def removeDCOffset(arr,state):
# This is to Remove The DC Offset By Using High Pass Filters
	hp_cutoff_Hz = 1.0 # cuttoff freq of 1 Hz (from 0-1Hz all the freqs at attenuated)
	b, a = signal.butter(2, hp_cutoff_Hz/(250 / 2.0), 'highpass')
	dcOutput, state = signal.lfilter(b, a, arr, zi=state)
	return dcOutput, state


Zstate = {}
Zstate['notch'] = {}
Zstate['dc_offset'] = {}
Zstate['bandpass'] = {}
for c in range(8):
	Zstate['notch'][c] = [0,0,0,0,0,0]
	Zstate['bandpass'][c] = [0,0,0,0,0,0]
	Zstate['dc_offset'][c] = [0,0]
sessions = os.listdir('./SessionData/')
data = np.zeros(shape=(1,250,8))
print("Retrieving Data")
for f in sessions:
	if("raghav" in f):
		for fi in os.listdir('./SessionData/'+f):
			data = np.vstack((data,np.load('./SessionData/'+f+'/'+fi)))
print(data.shape)
data = data[1:]
data = np.transpose(data,(2,0,1))
finalData = np.zeros(shape=(1,data.shape[1],250))
print("Applying filters to the RawData")
for c in range(data.shape[0]):
	# Create an empty numpy array and keep stacking it
	notchOut = np.zeros(shape=(1,250))
	print(c)
	for sample in data[c]:
		dcOutput, Zstate['dc_offset'][c] = removeDCOffset(sample,Zstate['dc_offset'][c])
		notchOutput, Zstate['notch'][c]  = notchFilter(dcOutput,Zstate['notch'][c])
		bandpassOutput, Zstate['bandpass'][c] = bandpass(notchOutput,Zstate['notch'][c])
		notchOut = np.vstack((notchOut,bandpassOutput.reshape(1,-1)))
		
	notchOut = notchOut[1:]
	finalData = np.vstack((finalData,np.expand_dims(notchOut,0)))
finalData = finalData[1:]

f = os.listdir('.')
label = []
print("Building Labels")
for k in f:
	if("232" in k):
		label = k
		break
l = np.load(label)

labelList = np.array([[0,0,0,int(i)+1,int(i)+1,int(i)+1] for i in l]).reshape(-1)

finalData = np.transpose(finalData,(1,0,2))

def trainer(data,labels):
	csp = CSP(n_components=3)
	for i,d in enumerate(data):
		data[i] = csp.fit_transform(d,[labels[i]]*8)	
	
	labels = np_utils.to_categorical(labels,3)
	print(data.shape)

	model = Sequential()
	model.add(Dense(160,activation='sigmoid',input_dim=3*250))
	model.add(Dense(160,activation='sigmoid'))
	model.add(Dense(160,activation='sigmoid'))
	model.add(Dense(3,activation='softmax'))

	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	model.summary()
	model.fit(data,labels,shuffle=True,epochs=20,batch_size=4,validation_split=0.3)
	pass

