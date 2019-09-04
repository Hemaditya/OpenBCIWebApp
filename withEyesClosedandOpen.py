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
f = open("model","w")

chunk_size=250
ds = app.DataStream(chunk_size=chunk_size)
plt.ion()
plt.show()

model = svm.SVC()

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


iterations = 1
r = 1
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
			#print("Dont Blink starts in 3: ")
			#time.sleep(1)
			#print("Dont Blink starts in 2: ")
			#time.sleep(1)
			#print("Dont Blink starts in 1: ")
			#time.sleep(1)
			pass
		else:
			#print("Blink starts in 3: ")
			#time.sleep(1)
			#print("Blink starts in 2: ")
			#time.sleep(1)
			#print("Blink starts in 1: ")
			#time.sleep(1)
			pass
		for i in range(iterations):	

		#while True:
			rawData = ds.read_chunk()
			c = 0
			rawData = rawData[0,:,6].reshape(-1)
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
		
	x = np.array(notch_0).reshape(-1)
	#y = np.array(notch_1).reshape(-1)
	x = x[np.where(np.abs(x) <= 400)]
	#y = y[np.where(np.abs(y) <= 400)]			
	freqs1, psd1 = signal.welch(x,250)
	#freqs2, psd2 = signal.welch(y,250)	
	#psd1, freqs1, t1 = mlab.specgram(x)
	#psd2, freqs2, t2 = mlab.specgram(y)

	alpha_idx = np.where(np.logical_and(freqs1>=8,freqs1<=14))
	#alpha_idx2 = np.where(np.logical_and(freqs2>=8,freqs2<=14))
#	print(alpha_idx2)
	alpha1 = psd1[alpha_idx]
	#alpha2 = psd2[alpha_idx2]
	b = np.average(alpha1.reshape(-1))
	#print(b)
	if(b >= 100):
		print("Noise")
	elif(b >= 10 and b <= 80):
		print("Eyes Close")
	elif(b < 10):
		print("Eyes Open")
	#print(b)
	#labels
	#model.fit(
	#if(b < 5.0):
	#	print("Eyes open")
	#else:
	#	print("Eyes closed ")
	#idx = np.where(np.logical_and(freqs1>=0, freqs1<=20))
	#ax1.cla()
	#ax2.cla()
	#ax1.set_ylim(0,100)
	#ax2.set_ylim(0,100)
	#ax1.plot(freqs1[idx],psd1[idx])
	#ax2.plot(freqs2[idx],psd2[idx])
	#x = raw_input("Save Data?: ")
	#t = time.strftime("%Y%m%d_%H%M%S")
	#if(x == 'y'):
	#	np.save('Data/'+t+"_eyes_open",x)
	#	np.save('Data/'+t+"_eyes_closed",y)
	#x = raw_input("Continue: ")
	#if(x == 'n'):
	#	break
os._exit(0)

