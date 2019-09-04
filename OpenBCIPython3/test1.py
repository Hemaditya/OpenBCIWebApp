import mneProgram as MNE
from sklearn.preprocessing import StandardScaler,normalize
import numpy as np
from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import time

files = MNE.retrieveFiles()
files = files["pickle"]
acc1 = 0
band =  [0,0]
a1 = range(30,12,-2)
a2 = range(0,1,1)
svm = SVC()
uVolts_per_count = (4.5)/24/(2**23-1)*1000000 # scalar factor to convert raw data into real world signal data
csp = CSP(n_components=8)
for i in a2:
	data,labels = MNE.dataFromFiles(files)
	data = data*uVolts_per_count
	print(data.shape)
	#for i, sample in enumerate(data):
	#	sc =StandardScaler()
	#	data[i] = sc.fit_transform(sample)
	#for i, sample in enumerate(data):
	#	data[i] = normalize(sample)
	data = MNE.applyFilters(data)
	indices = np.array(list(range(3,data.shape[0],6)))
	indices = np.asarray([[i,i+1,i+2] for i in indices]).reshape(-1)
	data = data[indices]
	labels = np.repeat(labels,3).reshape(-1)
	labels = labels.reshape(-1)
	for i, j in enumerate(labels):
		labels[i] = int(j)+1
	labels = labels.reshape(-1,3)
	zeros = np.zeros(shape=(labels.shape[0],3),dtype=np.int)
	labels = np.hstack((zeros,labels))
	labels = labels.reshape(-1)
	labels = labels[indices]
	#data1 = data[labels == '1']
	#data2 = data[labels == '2']
	#data = np.vstack((data1,data2))
	#labels1  = labels[labels == '1']
	#labels2  = labels[labels == '2']
	#labels = np.append(labels1,labels2)
	labels[labels == '1'] = 1
	labels[labels == '2'] = 1
	labels[labels == '3'] = 2
	labels[labels == '4'] = 2
	print(data.shape)
	print(labels.shape)
	print(np.unique(labels))

	X_train,X_test,Y_train,Y_test = train_test_split(data,labels,train_size=0.8,shuffle=True)
	classBalance = np.mean(Y_test == Y_test[0])
	print("CLASS BALANCE: ",max(classBalance, 1.0 - classBalance))
	X_train = csp.fit_transform(X_train,Y_train)
	X_test = csp.transform(X_test)
	
	X_train = svm.fit(X_train,Y_train)
	score = svm.predict(X_test)
	print(score)
	print(Y_test)
	print(np.mean(score==Y_test))

#for i in a2:
#	data,labels = MNE.dataFromFiles(files)
#	sc =StandardScaler()
#	#for i, sample in enumerate(data):
#	#	data[i] = normalize(sample)
#	data = MNE.applyFilters(data)
#	indices = list(range(3,data.shape[0],6))
#	indices = np.asarray([[i,i+1,i+2] for i in indices]).reshape(-1)
#	#indices = np.asarray([[i+1,i+2] for i in indices]).reshape(-1)
#	indices = np.asarray([[i] for i in indices]).reshape(-1)
#	data = data[indices]
#	labels = np.repeat(labels,3).reshape(-1)
#	#labels = np.repeat(labels,2).reshape(-1)
#	labels = labels.reshape(-1)
#	for i, j in enumerate(labels):
#		labels[i] = int(j)+1
#	zeros = np.zeros(shape=(60,3),dtype=np.int)
#	#zeros = np.zeros(shape=(,3),dtype=np.int)
#	labels = labels.reshape(-1,3)
#	#labels = labels.reshape(-1,2)
#	#labels = np.hstack((zeros,labels))
#	labels[labels == '1'] = 1
#	labels[labels == '2'] = 1
#	labels[labels == '3'] = 2
#	labels[labels == '4'] = 2
#	labels = labels.reshape(-1)
#	print(data.shape)
#	print(labels.shape)
#	print(np.unique(labels))
#
#	
#	classBalance = np.mean(labels == labels[0])
#	print("CLASS BALANCE: ",max(classBalance, 1.0 - classBalance))
#	svm = SVC()
#	csp = CSP(n_components=8)
#	X_train,X_test,Y_train,Y_test = train_test_split(data,labels,train_size=0.8,shuffle=True)
#	X_train = csp.fit_transform(X_train,Y_train)
#	X_test = csp.transform(X_test)
#	
#	X_train = svm.fit(X_train,Y_train)
#	score = svm.predict(X_test)
#	print(score)
#	print(Y_test)
#	print(np.mean(score == Y_test))
#
