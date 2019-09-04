import app
from sklearn.preprocessing import normalize
import pickle
from sklearn import svm
import numpy as np
import os
from scipy import signal
import torch
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn
import time
import matplotlib.mlab as mlab
import test5
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,normalize
from sklearn.linear_model import LogisticRegression as LG
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
#import trainer

#import timeEntropy as t


chunk_size=125
ds = app.OpenBCI(chunk_size=125,createSession=False)
plt.ion()
plt.show()
freq = np.fft.rfftfreq(chunk_size)*(250.0)
delta = np.where(np.logical_and(freq>=0, freq<4))
theta = np.where(np.logical_and(freq>=4, freq<8))
alpha = np.where(np.logical_and(freq>=8, freq<15))
beta = np.where(np.logical_and(freq>=15, freq<25))
gamma = np.where(np.logical_and(freq>=30,freq<=45))

#kfig, (ax) = plt.subplots(2,2)
#kax = ax.reshape(-1)
ss = []
l = []
model = []

def newRun():
	global model,ss,l
	import poo
	deita = poo.daata
	#deita[deita > 5000.0] = 5000.0
	#deita = deita/5000.0
	feta = deita
	print(feta.shape)
	train, test = train_test_split(feta,train_size=0.8) 
	print(train.shape)
	x_train, y_train = train[:,:,0:-1], train[:,0,-1] 
	x_test, y_test = test[:,:,0:-1], test[:,0,-1] 
	ss = StandardScaler() 
	x_train = ss.fit_transform(x_train.reshape(x_train.shape[0],-1)) 
	x_test = ss.transform(x_test.reshape(x_test.shape[0],-1)) 
	l = lda(n_components=3) 
	TRAIN = l.fit_transform(x_train,y_train.reshape(y_train.shape[0],-1)) 
	TEST = l.transform(x_test) 
	classifier = LG() 
	model = classifier.fit(TRAIN,y_train) 
	pred = model.predict(TEST) 
	cm = confusion_matrix(y_test,pred) 
	print(cm)

def bandpass(arr,state):
		# This is to allow the band of signal to pass with start frequency and stop frequency
		start = 5
		stop = 50
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
					

iterations = 5
r = 3
states = {0:"No Movement", 1:"Left hand Movement", 2:"Right Hand Movement"}
Zstate = {}
Zstate['notch'] = {}
Zstate['dc_offset'] = {}
Zstate['bandpass'] = {}
what_state = 0
channels = np.arange(8)
for c in range(8):
	Zstate['notch'][c] = [0,0,0,0,0,0]
	Zstate['bandpass'][c] = [0,0,0,0,0,0]
	Zstate['dc_offset'][c] = [0,0]

def run():
	global what_state,l,ss,model
	while True:
		notch_0 = []
		notch_1 = []
		rawData = {}
		for k in range(r):
			
			print(states[k]+" starts in 3")
			time.sleep(1)
			print(states[k]+" starts in 2")
			time.sleep(1)
			print(states[k]+" starts in 1")
			time.sleep(1)
			rawData[k] = []
			t = 150
			strip_times = 20
			channels_times = 5
			#for i in range(iterations):	
			while True:
				fftOuts = np.zeros(shape=(1,chunk_size))
				rawData[k] = ds.read_chunk()[0].reshape(chunk_size,8)
				fftT = np.zeros(shape=(1,chunk_size))
				for c in range(8):
					data = rawData[k][:,c].reshape(-1)
					dcOutput, Zstate['dc_offset'][c] = removeDCOffset(data,Zstate['dc_offset'][c])
					notchOutput, Zstate['notch'][c]  = notchFilter(dcOutput,Zstate['notch'][c])
					bandpassOutput, Zstate['bandpass'][c] = bandpass(notchOutput,Zstate['bandpass'][c])
					notchOutput = bandpassOutput
					fftT = np.vstack((fftT,notchOutput.reshape(1,chunk_size)))
				fftT = fftT[1:]	
				fftT = np.expand_dims(fftT,0)
				fftT = test5.csp.transform(fftT)
				pred = test5.svm.predict(fftT)
				#if(x == 0):
				#	print("no movement")
				#elif(x == 1):
				#	print("finger movement")
				#else:
				#	print("Hand clench")
				##elif(x == 2):
				##	print("artifact")
				print("OUPUT: ",pred)

					#wfftOuts = np.vstack((fftOuts,fftO))
				#fftOuts = fftOuts[1:]/20000.0
				#g = np.sum(fftOuts,0)
				#plt.clf()
				#plt.ylim(0,10000)
				#plt.plot(g[13:30])
				#plt.draw()
				#plt.pause(0.001)
				#fftOuts = fftOuts.reshape(1,-1)
				#fftOuts = ss.transform(fftOuts)
				#fftOuts = l.transform(fftOuts)
				##fftOuts = torch.from_numpy(fftOuts).float().unsqueeze(0)
				#out = model.predict(fftOuts)
				#print(out)

			#	print(torch.max(out,1)[1])
				#pl = np.roll(pl,-strip_times,0)	
				#dt = np.repeat(fftOuts[:,1:46],channels_times,-1)
				#pl[-strip_times:] = np.repeat(np.expand_dims(dt,0),strip_times,0)
				#plt.clf()
				#plt.imshow(np.flipud(np.transpose(pl.reshape(t*strip_times,8*45*channels_times),(1,0))),norm=None).set_clim(0,1.0)
				#plt.colorbar()
				#plt.pause(0.00001)
				#plt.draw()
			#
		for i in range(r):
			fftOuts[i] = {}
			for c in channels:
				fftOuts[i][c] = np.zeros(shape=(1,126))
				for j in range(iterations):
					data = rawData[i][j,:,c].reshape(-1)
					dcOutput, Zstate['dc_offset'][c] = removeDCOffset(data,Zstate['dc_offset'][c])
					notchOutput, Zstate['notch'][c]  = notchFilter(dcOutput,Zstate['notch'][c])
					no = notchOutput[np.where(np.abs(notchOutput) <= 400)]
					fftO = np.abs(np.fft.rfft(no,250)).reshape(1,126)
					fftOuts[i][c] = np.vstack((fftOuts[i][c],fftO))

				fftOuts[i][c] = fftOuts[i][c][1:]
				label = np.repeat(i,fftOuts[i][c].shape[0]).reshape(fftOuts[i][c].shape[0],1)
				fftOuts[i][c] = np.hstack((fftOuts[i][c],label))
				print(np.sum(fftOuts,1).shape)

		# plotting the data
		for j in ax:
			j.cla()


		print("Delta: Red")
		print("Theta: Blue")
		print("Alpha: Green")
		print("Beta: Yellow")
		print("Gamma: Black")
		for k in fftOuts.keys():
			d = fftOuts[k]
			for key in d:
				p = d[key]
				points = []
				points.append(np.average(p[:,delta],axis=2).reshape(-1))
				points.append(np.average(p[:,theta],axis=2).reshape(-1))
				points.append(np.average(p[:,alpha],axis=2).reshape(-1))
				points.append(np.average(p[:,beta],axis=2).reshape(-1))
				points.append(np.average(p[:,gamma],axis=2).reshape(-1))
				colors = ["red","blue","green","yellow","black"]
				for i, band in enumerate(points):
					ax[k].scatter(np.repeat(key,band.shape[-1]), band.reshape(-1), c=colors[i])

		plt.draw()

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
		data = ds.read_chunk(n_chunks=1)[0]
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
