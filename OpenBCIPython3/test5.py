import mneProgram as MNE
import trainers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix as cm
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from mne.decoding import CSP
from keras import Sequential
from keras.layers import Dense, Convolution1D, Dropout, AvgPool1D, Flatten,MaxPooling1D, Convolution2D
from keras.utils import np_utils
import pywt
files = MNE.retrieveFiles()
files = files["pickle"]
data,labels = MNE.dataFromFiles(files)
left = 0  # the left side of the subplots of the figure
right = 1  # the right side of the subplots of the figure
bottom = 0  # the bottom of the subplots of the figure
top = 1     # the top of the subplots of the figure
wspace = 0  # the amount of width reserved for space between subplots,
              # expressed as a fraction of the average axis width
hspace = 0  # the amount of height reserved for space between subplots,
              # expressed as a fraction of the average axis height

data = MNE.applyFilters(data)
labels [labels == '0'] = 1
labels[labels == '1'] = 1
labels[labels == '2'] = 1
labels[labels == '3'] = 1
labels = labels.repeat(3).reshape(-1,3)
zeros = np.zeros(shape=(labels.shape[0],3),dtype='int')
labels = np.hstack((zeros,labels)).reshape(-1)
indices = list(range(3,data.shape[0],12))
l = np.array([[i+j for i in range(3)] for j in indices]).reshape(-1)
#data = data[l]
print(data.shape,labels.shape)
csp = []
svm = []


left = 0  # the left side of the subplots of the figure
right = 1  # the right side of the subplots of the figure
bottom = 0  # the bottom of the subplots of the figure
top = 1     # the top of the subplots of the figure
wspace = 0  # the amount of width reserved for space between subplots,
              # expressed as a fraction of the average axis width
hspace = 0  # the amount of height reserved for space between subplots,

#for i in range(8):
#	plt.plot(data[:,4,:].reshape(-1))
