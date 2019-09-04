import app
import pickle
from sklearn import svm
import numpy as np
import os
from scipy import signal
import learner
import torch
import matplotlib.pyplot as plt
import pickle
import time
import matplotlib.mlab as mlab
#import trainer

#import timeEntropy as t

chunk_size=250
ds = app.DataStream(chunk_size=chunk_size)
plt.ion()
plt.show()
freq = np.fft.rfftfreq(chunk_size)*(250.0)
delta = np.where(np.logical_and(freq>=0, freq<4))
theta = np.where(np.logical_and(freq>=4, freq<8))
alpha = np.where(np.logical_and(freq>=8, freq<15))
beta = np.where(np.logical_and(freq>=15, freq<25))
gamma = np.where(np.logical_and(freq>=30,freq<=45))

fig, (ax) = plt.subplots(2,2)
ax = ax.reshape(-1)

def bandpass(arr,state):
		# This is to allow the band of signal to pass with start frequency and stop frequency
		start = 13
		stop = 25
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
					

iterations = 15
r = 2*4
states = {0:"No Movement", 1:"Left hand Movement", 2:"Right Hand Movement",3:"JJJA"}
Zstate = {}
Zstate['notch'] = {}
Zstate['dc_offset'] = {}
Zstate['bandpass'] = {}
what_state = 0
channels = np.arange(8)
for c in range(8):
	Zstate['notch'][c] = [0,0,0,0,0,0]
	Zstate['dc_offset'][c] = [0,0]
	Zstate['bandpass'][c] = [0,0,0,0,0,0]

def run():
	global what_state
	while True:
		notch_0 = []
		notch_1 = []
		rawData = {}
		fftOuts = {}
		for k in range(r):
			
			print(states[0]+" starts in 3")
			time.sleep(1)
			print(states[0]+" starts in 2")
			time.sleep(1)
			print(states[0]+" starts in 1")
			time.sleep(1)
			rawData[k] = []
			for i in range(iterations):	
				if((i+1) % 5 == 0 and i+1 > 0):
					print('MOVE')
				else:
					print(i+1)
				rawData[k].append(ds.read_chunk().reshape(chunk_size,8))
			rawData[k] = np.array(rawData[k])
			
		for i in range(r):
			fftOuts[i] = {}
			for c in channels:
				fftOuts[i][c] = np.zeros(shape=(1,chunk_size+1))
				for j in range(iterations):
						
					data = rawData[i][j,:,c].reshape(-1)
					dcOutput, Zstate['dc_offset'][c] = removeDCOffset(data,Zstate['dc_offset'][c])
					notchOutput, Zstate['notch'][c]  = notchFilter(dcOutput,Zstate['notch'][c])
					bandpassOutput, Zstate['bandpass'][c] = bandpass(notchOutput,Zstate['bandpass'][c])
					#no = notchOutput[np.where(np.abs(notchOutput) <= 400)]
					#fftO = np.abs(np.fft.rfft(no,250)).reshape(1,126)
					#fftOuts[i][c] = np.vstack((fftOuts[i][c],fftO))
					#notchOutput = dcOutput
					#notchOutput = bandpassOutput
					if(j+1 % 5 == 0):
						if(r == 0):
							notchOutput = np.append(notchOutput,1)
						if(r == 1):
							notchOutput = np.append(notchOutput,2)
					else:
						notchOutput = np.append(notchOutput,0)

					notchOutput = notchOutput.reshape(1,chunk_size+1)
					fftOuts[i][c] = np.vstack((fftOuts[i][c],notchOutput))



				fftOuts[i][c] = fftOuts[i][c][1:]
				#label = np.repeat(i,fftOuts[i][c].shape[0]).reshape(fftOuts[i][c].shape[0],1)
				#fftOuts[i][c] = np.hstack((fftOuts[i][c],label))
				#print(fftOuts.shape)

		# plotting the data
		#for j in ax:
		#	j.cla()


		#print("Delta: Red")
		#print("Theta: Blue")
		#print("Alpha: Green")
		#print("Beta: Yellow")
		#print("Gamma: Black")
		#for k in fftOuts.keys():
		#	d = fftOuts[k]
		#	for key in d:
		#		p = d[key]
		#		points = []
		#		points.append(np.average(p[:,delta],axis=2).reshape(-1))
		#		points.append(np.average(p[:,theta],axis=2).reshape(-1))
		#		points.append(np.average(p[:,alpha],axis=2).reshape(-1))
		#		points.append(np.average(p[:,beta],axis=2).reshape(-1))
		#		points.append(np.average(p[:,gamma],axis=2).reshape(-1))
		#		colors = ["red","blue","green","yellow","black"]
		#		for i, band in enumerate(points):
		#			ax[k].set_ylim(0,8000)
		#			ax[k].scatter(np.repeat(key,band.shape[-1]), band.reshape(-1), c=colors[i])

		#plt.draw()

		t = time.strftime("%Y%m%d_%H%M%S")
		x = raw_input("Save?: ")
		if(x != 'n'):
			with open("HandMovement/"+"FFTOUTPUT_"+t, 'w') as f:	
				pickle.dump(fftOuts,f)
		else:
			pass
		x = raw_input("Continue?: ")
		if(x == 'y'):
			continue
		else:
			os._exit(0)

def check_average_of_each_electrode(w):
	data = w[0]
	data = np.average(data,axis=0)
	return data

def check_electrode_connectivity(data):
	# Since there is only one chunk in each buffer use only the first element from data_buffer
	avg = check_average_of_each_electrode(data)
	ELECTRODE_THRESHOLD = 50000
	avg = (abs(avg) <= ELECTRODE_THRESHOLD)
	return avg


def checkElectrodeConnectivity():
	i = 0
	while(True):
		data = ds.read_chunk(ck=50)
		conn = check_electrode_connectivity(data)
		if(conn.all() == True):
			i = i+1
			print("All Connected")
		else:
			i = 0
			indices = np.array(np.nonzero(conn == False))
			print("Channels: ",np.array2string(indices,separator=',').strip('[]')," are not connected")
		if(i >= 15):
			break


#checkElectrodeConnectivity()
run()
