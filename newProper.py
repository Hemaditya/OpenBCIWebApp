import app
import pickle
from sklearn import svm
import numpy as np
import os
from scipy import signal
import learner
import torch
import matplotlib.pyplot as plt
model = learner.NN()
#model.load_state_dict(torch.load('myModel'))
model = torch.load('Artifact Removal')
model = model.double()

import time
import matplotlib.mlab as mlab
import trainer

chunk_size=250
ds = app.DataStream(chunk_size=chunk_size)
plt.ion()
plt.show()

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

def pureFFT(arr,NFFT=256,chunk_size=250):
	fftOutput = abs(np.fft.rfft(arr))
	freqs = np.fft.rfftfreq(chunk_size)
	return (freqs,fftOutput)


iterations = 5
r = 2
Zstate = {}
Zstate['notch'] = {}
Zstate['dc_offset'] = {}
for c in range(8):
	Zstate['notch'][c] = [0,0,0,0,0,0]
	Zstate['dc_offset'][c] = [0,0]
notchO = []
print("The recording starts in 3")
time.sleep(1)
print("The recording starts in 2")
time.sleep(1)
print("The recording starts in 1")
time.sleep(1)
#for i in range(iterations):	
while True:
	notch_0 = []
	notch_1 = []
	for k in range(r):
		model = torch.load('Artifact Removal').double()
		if((k%2) == 0):
			print("Dont Blink starts in 3: ")
			time.sleep(1)
			print("Dont Blink starts in 2: ")
			time.sleep(1)
			print("Dont Blink starts in 1: ")
			time.sleep(1)
		else:
			print("Blink starts in 3: ")
			time.sleep(1)
			print("Blink starts in 2: ")
			time.sleep(1)
			print("Blink starts in 1: ")
			time.sleep(1)
		for i in range(iterations):	

		#while True:
			rawData = ds.read_chunk()
			c = 0
			rawData = rawData[0,:,0].reshape(-1)
			dcOutput, Zstate['dc_offset'][c] = removeDCOffset(rawData,Zstate['dc_offset'][c])
			notchOutput, Zstate['notch'][c]  = notchFilter(dcOutput,Zstate['notch'][c])
			if((k%2) == 0):
				notch_0.append(notchOutput)
			else:
				notch_1.append(notchOutput)
			#spec = mlab.specgram(notchOutput)[0]
			#spec = torch.from_numpy(notchOutput).reshape(1,250,1)
			#model = model.double()
			#x = model(spec).argmax().item()
			#if(x == 0):
			#	print("No Blink")
			#elif(x == 1):
			#	print("Blink")
			#elif(x == 2):
			#	print("Noise")
		
	x = np.hstack((np.array(notch_0).reshape(-1,chunk_size),np.zeros(shape=(iterations*r/2,1))))
	y = np.hstack((np.array(notch_1).reshape(-1,chunk_size),np.ones(shape=(iterations*r/2,1))))
	d = np.vstack((x,y))
	#f = open("MODEL","r")
	#print(pickle.load(f).predict(d[:,0:-1]))
	#f.close()
	out = model(torch.from_numpy(d[:,0:-1]).reshape(-1,250,1))
	o = np.argmax(out.detach().numpy(),axis=1)
	dataPoints = []	
	label = []
	plt.clf()
	plt.plot(d[:,0:-1].reshape(-1))
	for i, sample in enumerate(d):
		if(np.abs(sample).reshape(-1).max() >= 400):
			sample[-1] = 2
			d[i][-1] = 2
		label.append(sample[-1])
		plt.axvline(i*chunk_size,color="red")	
	#print(label)
	print(o)
	print(label)
	label = np.array(label)	
	print(np.sum(o.reshape(-1) - label.reshape(-1))/float(o.reshape(-1).shape[0]))
	x = raw_input("Input: ")
	if(x == "y" or x == 't'):
		t = time.strftime("%Y%m%d_%H%M%S_notchData")
		np.save("Data/"+t,d)
		if(x == 't'):
			model = trainer.train(d)
	x = raw_input("Continue: ")
	if(x == 'n'):
		break
	elif(x == 'y'):
		continue
	
os._exit(0)

