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

import time
import matplotlib.mlab as mlab
#import trainer

#import timeEntropy as t

chunk_size=250
ds = app.DataStream(chunk_size=chunk_size)
plt.ion()
plt.show()
model = torch.load('/home/hemaditya/BCI/axon/OpenBCIWebApp/Flask/app/modules/brand')
model.eval()

#fig, (ax1,ax2) = plt.subplots(2,1)
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
					

iterations = 1
r = 1
Zstate = {}
Zstate['notch'] = {}
Zstate['dc_offset'] = {}
what_state = 2
jaa_state = 0
for c in range(8):
	Zstate['notch'][c] = [0,0,0,0,0,0]
	Zstate['dc_offset'][c] = [0,0]

def run():
	global what_state,jaa_state
	while True:
		notchO = []
		notch_0 = []
		notch_1 = []
		dc1 = []
		nO = []
		for k in range(r):
			if((k%2) == 0):
				pass
			else:
				pass
			for i in range(iterations):	
				rawData = ds.read_chunk()
				c = 0
				newRaw = rawData[0,:,0].reshape(-1)
				rawData = rawData[0,:,0].reshape(-1)
					
				prevState1 = Zstate['dc_offset'][c]
				prevState2 = Zstate['notch'][c]
				dc1, x = removeDCOffset(newRaw,prevState1)
				nO, y = notchFilter(dc1,prevState2)

				dcOutput, Zstate['dc_offset'][c] = removeDCOffset(rawData,Zstate['dc_offset'][c])
				notchOutput, Zstate['notch'][c]  = notchFilter(dcOutput,Zstate['notch'][c])
				if((k%2) == 0):
					notch_0.append(notchOutput)
				else:
					notch_1.append(notchOutput)
				
		x = np.array(notch_0).reshape(-1)
		x = x[np.where(np.abs(x) <= 400)]
		nO = nO.reshape(-1)
		nO = nO[np.where(np.abs(nO) <= 400)]
		freqs1, psd1 = signal.welch(x,250)
		out,f,t = mlab.specgram(nO.reshape(-1),NFFT=250)	
		h = np.average(out[0:20].reshape(-1))
		fft = np.abs(np.fft.rfft(nO,chunk_size)[0:10])
		out = model(torch.from_numpy(fft.reshape(1,10,1))).argmax().item()
		print(out)
		if(out == 1):
			jaa_state = 1
		else:
			jaa_state = 0
		alpha_idx = np.where(np.logical_and(freqs1>=8,freqs1<=14))
		alpha1 = psd1[alpha_idx]
		b = np.average(alpha1.reshape(-1))
		if(b >= 100):
			what_state = 2
			#print("Noise")
			
		elif(b >= 10 and b <= 80):
			what_state = 1
			#print("EYES CLOSED")

		elif(b < 10):
			what_state = 0
			#print("EYES OPEN")

