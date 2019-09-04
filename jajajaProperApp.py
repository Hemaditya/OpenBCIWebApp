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
model = torch.load("HA").double()
model.eval()

import time
import matplotlib.mlab as mlab
#import trainer

#import timeEntropy as t

chunk_size=250
ds = app.DataStream(chunk_size=chunk_size)
plt.ion()
plt.show()



fig, (ax1,ax2) = plt.subplots(2,1)
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

def timeEntropy(N,m,r,arr):
	component = []
	for i in range(N-m+1):
		d = arr[i:i+m]
		sim = []
		for j in range(N-m+1):
			if(j != i):	
				if(distance(d,arr[j:j+m])<=r):
					sim.append(1)
				else:
					sim.append(0)
			else:
				pass
		ci = sum(sim)/float((len(sim)))
		component.append(ci)
	
	amr = (1.0/float(N-m+1)) * sum([np.log(val) for val in component])
					

iterations = 3
r = 3
states = {0:"No Movement", 1:"Left hand Movement", 2:"Right Hand Movement"}
Zstate = {}
Zstate['notch'] = {}
Zstate['dc_offset'] = {}
what_state = 0
for c in range(8):
	Zstate['notch'][c] = [0,0,0,0,0,0]
	Zstate['dc_offset'][c] = [0,0]

def run():
	global what_state
	while True:
		notchO = []
		notch_0 = []
		notch_1 = []
		for k in range(r):
			print(state[k]+" starts in 3")
			time.sleep(1)
			print(state[k]+" starts in 2")
			time.sleep(1)
			print(state[k]+" starts in 1")
			time.sleep(1)

			rawData = []
			for i in range(iterations):	
				rawData = ds.read_chunk()
				c = 0
				rawData = rawData[0,:,0].reshape(-1)
				dcOutput, Zstate['dc_offset'][c] = removeDCOffset(rawData,Zstate['dc_offset'][c])
				notchOutput, Zstate['notch'][c]  = notchFilter(dcOutput,Zstate['notch'][c])
				no = notchOutput[np.where(np.abs(notchOutput) <= 400)]
				fftO = np.abs(np.fft.rfft(no,250))[0:10].reshape(-1)
				fftO = fftO/20000.0
				print(model(torch.from_numpy(fftO)).argmax().item())
				if((k%2) == 0):
					notch_0.append(notchOutput)
				else:
					notch_1.append(notchOutput)
				
		ax1.cla()
		ax2.cla()
		x = np.array(notch_0).reshape(-1,250)
		y = np.array(notch_1).reshape(-1,250)
		print(x.shape)
		print(y.shape)
		
		fftX = np.zeros(shape=(1,126))
		fftY = np.zeros(shape=(1,126))
		freqs = np.fft.rfftfreq(250)*250
		for sample in x:
			sample = sample[np.where(np.abs(sample) <= 400)]
			fftX = np.vstack((fftX, np.abs(np.fft.rfft(sample,250).reshape(1,126))))
			ax1.plot(freqs[0:10],fftX[-1,0:10])


		for sample in y:
			sample = sample[np.where(np.abs(sample) <= 400)]
			fftY = np.vstack((fftY, np.abs(np.fft.rfft(sample,250).reshape(1,126))))
			ax2.plot(freqs[0:10],fftY[-1,0:10])
		
		
		fftX = np.hstack((fftX[1:,0:10],np.zeros(shape=(fftX.shape[0]-1,1))))
		fftY = np.hstack((fftY[1:,0:10],np.ones(shape=(fftY.shape[0]-1,1))))
		fftFull = np.vstack((fftX,fftY))
		plt.draw()
		t = time.strftime("%Y%m%d_%H%M%S")
		t = 'JAJAJA'
		x = raw_input("Save?: ")
		if(x == 'y'):
			np.save("FFTOUT",fftFull)
		else:
			pass
		x = raw_input("Continue?: ")
		if(x == 'y'):
			continue
		else:
			os._exit(0)
while True:
	run()
