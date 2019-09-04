import mneProgram as MNE
import pywt
import matplotlib.pyplot as plt
from scipy.signal import spectrogram 
from sklearn.preprocessing import StandardScaler,normalize
import model
import numpy as np
from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import time
import keras 
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils

files = MNE.retrieveFiles()
files = files["pickle"]
acc1 = 0
band =  [0,0]
a1 = range(30,12,-2)
a2 = range(0,1,1)
mean = []
std = []
data,labels = MNE.dataFromFiles(files)
for i, j in enumerate(labels):
	labels[i] = int(j)+1
labels[labels == '1'] = 0
labels[labels == '2'] = 1
labels[labels == '3'] = 0
labels[labels == '4'] = 1
classBalance = np.mean(labels == labels[0])
print("CLASS BALANCE: ",max(classBalance, 1.0 - classBalance))
uVolts_per_count = (4.5)/24/(2**23-1)*1000000 # scalar factor to convert raw data into real world signal data
indices = list(range(3,data.shape[0],6))
indices = np.asarray([[i,i+1,i+2] for i in indices]).reshape(-1)
data = data[indices]
labels = labels.reshape(-1)
labels = np.repeat(labels,3).reshape(-1)
i = labels[labels == 0]
#data = data*uVolts_per_count
data = MNE.applyFilters(data)
data = data.transpose(1,0,2)
print(data.shape)
fig, ax = plt.subplots(2,1)
#print(np.average(data))
#for i in [0,1]:
#	ax[i].plot(data[i+3].reshape(-1))
#	ax[i].set_ylim(-500,500)
spec = np.abs(np.fft.rfft(data))
print(spec.shape)
spec = 10*np.log10(spec)
spec = spec.transpose(1,0,2)
spec = spec[:,:,:40]
spec = (spec - np.mean(spec))/np.std(spec)
spec = np.expand_dims(spec,-1)
print(spec.shape)
labels = np_utils.to_categorical(labels,2)

print(labels.shape)
model.model.fit(spec,labels,epochs=50,validation_split=0.3,shuffle=True)
