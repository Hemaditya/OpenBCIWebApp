import mneProgram as MNE
import bandpower as b
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Retrieve 2 different data
files = MNE.retrieveFiles()['pickle']
files.sort()
print(files)
data, labels = MNE.dataFromFiles(files)
data2, labels2 = [],[]
k = 2
if(k == 2):
	files = MNE.retrieveFiles()['pickle']
	files.sort()
	data2, labels2 = MNE.dataFromFiles(files)


fdata = MNE.applyFilters(data)
fdata2 = MNE.applyFilters(data2)

def calc_bandpower(data):
	'''
		Input: data.shape = (chunks,channel,chunk_size)
		Output will be shape of (channels,chunks,5bands)
	'''

	band = {}
	band['delta'] = [1,4]
	band['theta'] = [5,8]
	band['alpha'] = [9,13]
	band['beta'] = [12,30]
	band['gamma'] = [30,50]

	data = data.transpose(1,0,2)
	
	chanD = []
	for i,c in enumerate(data):
		print("Processing channel: ",i)
		chunkD = []
		for j, s in enumerate(c):
			print("Processing chunk: ",j,"/",c.shape[0])
			bandPow = []
			for k in band:
				bp = b.bandpower(s,250.0,band[k],1)
				bandPow.append(bp)
			chunkD.append(bandPow)
		chanD.append(chunkD)
	
	return np.asarray(chanD)

def bandPow(data):
	'''
		Calculate bandpower of entire 20 trials
	'''

	band = {}
	band['delta'] = [1,4]
	band['theta'] = [4,8]
	band['alpha'] = [8,14]
	band['beta'] = [16,30]
	band['gamma'] = [30,50]
	data = data.transpose(1,0,2)
	data = data.reshape(data.shape[0],-1)
	chanD = []
	for i,c in enumerate(data):
		bandP = []
		for k in band:
			bp = b.bandpower(c,250.0,band[k],2)
			bandP.append(bp)
		chanD.append(bandP)
	
	return np.asarray(chanD)

def regressor(out):
	
	outs = np.array([-0.0651,-0.0136,0.0256,-0.0072,0.0009,-0.0032])

	out = np.append(1,out)

	val = np.dot(outs,out)
	
	return abs(val)

def chad():
	chanDa = [0,2,6,7]
	d1 = fdata[:120,chanDa,:]
	#d2 = fdata[120:240,chanDa,:]
	#d3 = fdata[240:,chanDa,:]
	d1 = d1[2:]
	#d2 = d2[2:]
	#d3 = d3[2:]
	out1 = bandPow(d1)
	#out2 = bandPow(d2)
	#out3 = bandPow(d3)
	out1 = out1.mean(axis=0)
	#out2 = out2.mean(axis=0)
	#out3 = out3.mean(axis=0)
	print(out1)
	#print(out2)
	#print(out3)
	r1 = regressor(out1)
	#r2 = regressor(out2)
	#r3 = regressor(out3)
	print(r1)	
	#print(r2)	
	#print(r3)	

def every5secs():
	chanDa = [0,2,6,7]
	d1 = fdata[:120,chanDa,:]
	d2 = fdata2[120:240,chanDa,:]
	d3 = fdata2[240:,chanDa,:]
	
	out1 = getBandPow(d1)
	out2 = getBandPow(d2)
	out3 = getBandPow(d3)
	fig, ax = plt.subplots(2,2)
	ax = ax.reshape(-1)
	for i in range(4):
		ax[i].plot(out1[:,i,1].reshape(-1),c='r')
		ax[i].plot(out2[:,i,1].reshape(-1),c='b')
		ax[i].plot(out3[:,i,1].reshape(-1),c='y')
	out1 = out1.mean(axis=1)
	out2 = out2.mean(axis=1)
	out3 = out3.mean(axis=1)

	print(out1.shape)
	print(out2.shape)
	print(out3.shape)

	l1 = getRegOut(out1)
	l2 = getRegOut(out2)
	l3 = getRegOut(out3)
	l1 = l1[1:]	
	l2 = l2[1:]	
	l3 = l3[1:]	
	print(l1.mean())
	print(l2.mean())
	print(l3.mean())
	plt.show()

def getRegOut(out):
	l = []
	for i in out:
		l.append(regressor(i))
	return np.asarray(l)

def getBandPow(data):
	indices = np.arange(0,120,5)
	out = []
	for i in indices:
		out.append(bandPow(data[i:i+5]))
	
	return np.asarray(out)

every5secs()
