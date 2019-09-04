import mne
from scipy import signal
import time
from mne.channels import read_layout
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from mne.decoding import CSP
import pickle
from mne.io import RawArray
import os
import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.layers import Dense,Dropout
from keras.utils import np_utils
from sklearn.preprocessing import normalize
chunk_size = 250
homeDir = './SessionData'

def retrieveFiles():
	x = input("Enter the user data you wanna collect: ")
	x = x.lower()
	return getFiles(x)

def getFiles(user):
	'''
		This function will retrieve all the files from the 
		folder for the $user
	'''

	# Convert the user to lowercase
	user = user.lower()
	
	# Check if the folder for that user exists
	if(os.path.isdir(homeDir+"/"+user) != True):
		print("The user name given is invalid and doesnt exist")
		return 0
	
	# If the folder exists then return all the files present in that folder	
	numpyFiles = []
	pickleFiles = []
	for f in os.listdir(homeDir+"/"+user):
		if(".npy" in f):
			numpyFiles.append(homeDir+"/"+user+"/"+f)
		elif(".pickle" in f):
			pickleFiles.append(homeDir+"/"+user+"/"+f)	
	
	return {"npy":numpyFiles,"pickle":pickleFiles}

def removeDCOffset(arr,state):
# This is to Remove The DC Offset By Using High Pass Filters
	hp_cutoff_Hz = 1.0 # cuttoff freq of 1 Hz (from 0-1Hz all the freqs at attenuated)
	b, a = signal.butter(2, hp_cutoff_Hz/(250 / 2.0), 'highpass')
	dcOutput, state = signal.lfilter(b, a, arr, zi=state)
	return dcOutput, state

		
def rawArrayToObject(data, channel_names='default', sfreq=250):
	'''
		Builds an MNE object from raw array
	'''
	
	'''
	  	The below 'default' config for channel names if taken from the paper 'Classification of multiple motor imagery using deep 
		convolutional neural networks and spatial filters' by Brenda E. Olivas-Padilla , Mario I. Chacon-Murguia
	'''

	if(channel_names=='default'):
		channel_names = ['FC1','FC2','C3','Cz','C4','CP1','CP2','Pz']
	channels = len(channel_names)

	# The data should be of the form (chunks,channels,chunk_size)
	shape = data.shape
	if(shape[1] != channels):
		print("Inconsistent channels in data and channel names")
		return 0
	
	# Convert the data to shape: (channels,num_of_data_points)
	data = np.transpose(data,(1,0,2))
	data = data.reshape(channels,-1)

	# Create info object
	ch_types = ['eeg']*channels
	info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types=ch_types)
	mneObject = mne.io.RawArray(data, info)

	return mneObject

def buildEpochs(mneObject, labels, policy=0):
	'''
		This function will  build epochs for the mneObject and labels
		policy referes to how labels should be assigned
	'''

	description = []
	onset = []
	n_epochs = 0
	if(policy == 0):
		description = [[0,0,0,int(i)+1,int(i)+1,int(i)+1] for i in labels]
		description = np.asarray(description).reshape(-1)
		n_epochs = description.shape[0]
		onset = np.arange(n_epochs)
	if(policy == 1):
		description = [[int(i)+1,int(i)+1,int(i)+1] for i in labels]	
		description = np.asarray(description).reshape(-1)
		n_epochs = description.shape[0]
		onset = np.arange(n_epochs)

	annotObject = mne.Annotations(onset,[1]*n_epochs,description=description)

	mneObject.set_annotations(annotObject)
	
	events,event_id = mne.events_from_annotations(mneObject)
	print(events)

	epochs = mne.Epochs(mneObject,events=events,event_id=event_id)

	return epochs


def dataFromPickle(filename):
	
	with open(filename,'rb') as f:
		data = pickle.load(f)


	'''
		The data will be in the format of dictionary with keys: raw, labels
		raw: Represents raw data
		labels: labels for the rawData
	'''
	return data

def removeDCOffset(arr,state):
# This is to Remove The DC Offset By Using High Pass Filters
	hp_cutoff_Hz = 1.0 # cuttoff freq of 1 Hz (from 0-1Hz all the freqs at attenuated)
	b, a = signal.butter(2, hp_cutoff_Hz/(250 / 2.0), 'highpass')
	dcOutput, state = signal.lfilter(b, a, arr, zi=state)
	return dcOutput, state

def cspSVMPipeline(epochs):
	'''
		This function will build a CSP and SVM pipeline for the epochs and then train it on that data
		Here the data retrieved from the epochs might not be same as the one from original data,
		since the mne package epochs automatically drops the bad data
	'''

	# Build a csp and svm layer
	csp = CSP(n_components=3)
	svm = SVC(verbose=False)
	lda = LDA()
	clf = Pipeline([('CSP',csp),('SVM',svm)])
	ldaOnly = Pipeline([('lda',lda)])
	svcOnly = Pipeline([('SVM',svm)])


	# Get the data and labels
	epochs_data = epochs.get_data() 
	labels = epochs.events[:,-1]


	# Create a cross-validation split
	cv = ShuffleSplit(10,test_size=0.2,random_state=42)

	method="SVM"

	# retrieve the scoreW
	if(method == "SVM"):
		epochs_data = epochs_data.reshape(epochs_data.shape[0],-1)
		scores = cross_val_score(svcOnly,epochs_data,labels,cv=cv,n_jobs=1)
	else:
		scores = cross_val_score(clf,epochs_data,labels,cv=cv,n_jobs=1)
	class_balance = np.mean(labels == labels[0])
	class_balance = max(class_balance, 1. - class_balance)
	print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
															  class_balance))

	return np.mean(scores)

def denseModel(mneEpochs, channel=None):
	'''
		This will create a dense model for the dataset
	'''
	epochs = 20
	epochsData = mneEpochs.get_data()
	labels = mneEpochs.events[:,-1] 
	if(channel == None):
		epochsData = epochsData.reshape(epochsData.shape[0],-1)
	else:
		#epochsData = epochsData[:,channel,:].reshape(epochsData.shape[0],-1)
		epochsData = epochsData[:,channel,:]
	

	# Applyin csp
	csp = CSP(n_components=8)
	epochsData = csp.fit_transform(epochsData,labels)
	plotChannel(epochsData,labels)
	print(epochsData.shape)

	# This is preprocessing step
	#epochsData = normalize(epochsData)
	labels = np_utils.to_categorical(labels)

	# Start creating the model
	model = Sequential()
	model.add(Dense(100,activation='relu',input_dim=2))
	model.add(Dropout(0.5))
	model.add(Dense(160,activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(160,activation='relu'))
	model.add(Dense(labels.shape[-1],activation='softmax'))

	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

	model.fit(epochsData,labels,batch_size=4,epochs=epochs,validation_split=0.2,shuffle=True)

def findAverage(mneEpochs):
	'''
		Find the average of 3 channels per sample
	'''

	epochs_data = mneEpochs.get_data() # will give data of shape(n_chunks,n_channels,chunk_size)
	
	chan_left = epochs_data[:,[1,3,6],:]
	chan_right = epochs_data[:,[2,5,7],:]
	
	chan_left = np.average(chan_left,0)
	chan_right = np.average(chan_right,0)
	fig, ax = plt.subplots(1,2)
	for i in chan_left:
		ax[0].plot(i)
	for i in chan_right:
		ax[1].plot(i)
	plt.show()

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

def applyFilters(data):
	'''	
		Input - The data will be in the shape of (n_chunks, channels,chunk_size)
		Output - (n_chunks, channels, chunk_size)
	'''

	# COnvert the data
	Zstate = {}
	Zstate['notch'] = {}
	Zstate['dc_offset'] = {}
	Zstate['bandpass'] = {}
	for c in range(8):
		Zstate['notch'][c] = [0,0,0,0,0,0]
		Zstate['bandpass'][c] = [0,0,0,0,0,0]
		Zstate['dc_offset'][c] = [0,0]
	data = data.transpose(1,0,2)
	finalData = np.zeros(shape=(1,data.shape[1],data.shape[-1]))
	for c in range(data.shape[0]):
		# Create an empty numpy array and keep stacking it
		notchOut = np.zeros(shape=(1,data.shape[-1]))
		print(c)
		for sample in data[c]:
			dcOutput, Zstate['dc_offset'][c] = removeDCOffset(sample,Zstate['dc_offset'][c])
			notchOutput, Zstate['notch'][c]  = notchFilter(dcOutput,Zstate['notch'][c])
			bandpassOutput, Zstate['bandpass'][c] = bandpass(notchOutput,Zstate['bandpass'][c])
			notchOut = np.vstack((notchOut,bandpassOutput.reshape(1,-1)))
			#notchOut = np.vstack((notchOut,notchOutput.reshape(1,-1)))
		
		notchOut = notchOut[1:]
		finalData = np.vstack((finalData,np.expand_dims(notchOut,0)))

	finalData = finalData[1:]
	finalData = finalData.transpose(1,0,2)
	return finalData

def plotChannel(epochs_data,labels):
	channel_1 = epochs_data[:,0]
	channel_2 = epochs_data[:,1]
	print(channel_1)
	print(channel_2)
	u = np.unique(labels)
	print(labels)
	label_a = {}
	label_b = {}
	for l in u:
		label_a[l] = []
		label_b[l] = []
	for i,j in enumerate(channel_1):
		label_a[labels[i]].append(channel_1[i])
		label_b[labels[i]].append(channel_2[i])

	fig,ax = plt.subplots(u.shape[0],2)

	for i,x in enumerate(label_a):
		#ax[i,0].set_xlim(0,1000)
		ax[i,0].plot(np.array(label_a[x]).reshape(-1))

	for i,x in enumerate(label_b):
		#ax[i,1].set_xlim(0,1000)
		#ax[i,1].set_ylim(-5000,5000)
		ax[i,1].plot(np.array(label_b[x]).reshape(-1))

	plt.show()
	pass	

def dataFromFiles(filenames):
	''' 
		This function will take in the paths of pickle files and then 
		build numpy array objects of them
	'''
	dataObj = []
	for f in filenames:
		dataObj.append(dataFromPickle(f))	
	

	data = np.zeros(shape=(1,8,chunk_size))
	labels = []
	for i in dataObj:
		d = i['raw']
		if(type(d) == tuple):
			d = d[0]
		print(type(d))
		d = d.transpose(0,2,1)
		data = np.vstack((data,d))
		if(type(labels) == list):
			labels = i['labels'].reshape(-1)
		else:
			labels = np.append(labels,i['labels'])
	data = data[1:]
	labels = np.asarray(labels).reshape(-1)
	print("LABELS: ",labels.shape)
	return (data,labels)
