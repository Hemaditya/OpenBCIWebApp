pos = ['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8','T3','T4','T5','T6','FZ','CZ','PZ']
import scipy.io as sio
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
matfile = sio.loadmat('matProp.mat')
left_data = np.copy(matfile['left'])
right_data = np.copy(matfile['right'])

left_data = left_data[:,150:]
left_data = left_data[:,:-150]
right_data = right_data[:,150:]
right_data = right_data[:,:-150]
left_data = left_data.reshape(left_data.shape[0],-1,1000)
right_data = right_data.reshape(right_data.shape[0],-1,1000)

fig,ax = plt.subplots(4,5)
ax = ax.reshape(-1)
plt.subplots_adjust(wspace=0.5,hspace=1.0)

Zstate = {}
Zstate['notch'] = {}
Zstate['dc_offset'] = {}
Zstate['bandpass'] = {}
for c in range(19):
	Zstate['notch'][c] = [0,0,0,0,0,0]
	Zstate['bandpass'][c] = [0,0,0,0,0,0]
	Zstate['dc_offset'][c] = [0,0]

def bandpass(arr,state):
		# This is to allow the band of signal to pass with start frequency and stop frequency
		start = 13
		stop = 30
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

left_channel_data = {}
right_channel_data = {}

for c in range(left_data.shape[0]):
	left_channel_data[c] = np.zeros(shape=(1,1000))
	for data in left_data[c]:
		dcOutput, Zstate['dc_offset'][c] = removeDCOffset(data.reshape(-1),Zstate['dc_offset'][c])
		notchOutput, Zstate['notch'][c]  = notchFilter(dcOutput,Zstate['notch'][c])
		bandpassOutput, Zstate['bandpass'][c] = bandpass(notchOutput,Zstate['bandpass'][c])
		left_channel_data[c] = np.vstack((left_channel_data[c],bandpassOutput.reshape(1,-1)))
	left_channel_data[c] = left_channel_data[c][1:]
print(left_channel_data[0].shape)

Zstate = {}
Zstate['notch'] = {}
Zstate['dc_offset'] = {}
Zstate['bandpass'] = {}
for c in range(19):
	Zstate['notch'][c] = [0,0,0,0,0,0]
	Zstate['bandpass'][c] = [0,0,0,0,0,0]
	Zstate['dc_offset'][c] = [0,0]

for c in range(right_data.shape[0]):
	right_channel_data[c] = np.zeros(shape=(1,1000))
	for data in right_data[c]:
		dcOutput, Zstate['dc_offset'][c] = removeDCOffset(data.reshape(-1),Zstate['dc_offset'][c])
		notchOutput, Zstate['notch'][c]  = notchFilter(dcOutput,Zstate['notch'][c])
		bandpassOutput, Zstate['bandpass'][c] = bandpass(notchOutput,Zstate['bandpass'][c])
		right_channel_data[c] = np.vstack((right_channel_data[c],bandpassOutput.reshape(1,-1)))
	right_channel_data[c] = right_channel_data[c][1:]
print(right_channel_data[0].shape)

left_log_energy = {}
right_log_energy = {}

for c in left_channel_data:
	allData = left_channel_data[c]
	left_log_energy[c] = []
	for data in allData:
		sqr = np.square(data.reshape(-1))
		Sum = np.sum(sqr)
		log = 10*np.log(Sum)
		left_log_energy[c].append(log)

for c in right_channel_data:
	allData = right_channel_data[c]
	right_log_energy[c] = []
	for data in allData:
		sqr = np.square(data.reshape(-1))
		Sum = np.sum(sqr)
		log = 10*np.log(Sum)
		right_log_energy[c].append(log)


for i in range(19):
	ax[i].hist(left_log_energy[i],alpha=0.7)
	ax[i].hist(right_log_energy[i],alpha=0.7)
	ax[i].title.set_text(pos[i])
plt.show()
