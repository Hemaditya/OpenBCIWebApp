import mneProgram as MNE
import pandas as pd
from pathlib import Path
import bandpower as b
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import sys

# Check if there are any command line argumnets
x = ""
if(len(sys.argv) == 2):
	x = sys.argv[-1]	
else:
	x = input("Enter the session name: ")
x = x.lower()
files = MNE.retrieveFiles(x)['pickle']
files.sort()
print(files)
data, labels = MNE.dataFromFiles(files)
data = MNE.applyFilters(data)

def preprocessData(data,channels=[0,2,6,7]):
	'''
		Preprocess the data by removing bad samples 
		and trimming the data to remove unwanted samples
	'''
	
	# First select channels to be processed
	data = data[:,channels,:]
	# Find the number of bad samples
	bad_samples = np.abs(data.reshape(data.shape[0],-1).max(axis=1)) > 200
	bad_samples_index = np.where(bad_samples == True)
	print("Bad Samples:", bad_samples.sum())
	print("Bad Samples Index:",bad_samples_index)
	# Remove bad samples
	newData = np.delete(data,bad_samples_index,axis=0)
	# To make the shape divisble by 5
	unwanted_samples = newData.shape[0]%10
	newData = newData[:-unwanted_samples]
	print("Data after removing bad and unwanted samples:",newData.shape)
	return newData

def processData(data,num=5,channels=[0,2,6,7],discard_first2=False):
	'''
		Input data should be of the shape (chunks,channels,chunk_size)
		Split the data into num splits and calculate the bandpower

		num - The number of splits you want to split the data into
	'''
	split = int(data.shape[0]/num)
	print("Old data shape: ",data.shape)
	data = data.reshape(num,split,data.shape[1],data.shape[2])
	print("New data shape: ",data.shape)
	
	# Discard the first 2 samples
	if(discard_first2 == True):
		data = data[:,2:,:]
	# Now calculate bandpower for each of the split
		
	print("Calculating Band power for each of the split")
	bandPower = []
	for dataSplit in data:
		bandPower.append(bandPow(dataSplit))
	
	return np.asarray(bandPower)

def processDataInChunks(data,split=5):
	'''
		Input data should be of the shape (chunks,channels,chunk_size)
		splits is the number of chunks for which the data should be split
	'''

	try:
		data = data.reshape(-1,split,data.shape[1],data.shape[2])
	except:
		print("The data is not splittable in as per the given split")
		return -1
	
	print("The data has been split successfully")

def bandPow(data):
	'''
		Input data shape should be (chunks,channels,chunk_size)
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

def saveDataToCSV(bandPower):
	''' 
		Input should be (chunks,channels,number_of_freq_bands)
		Take the bandPower and calculate the regressor output
	'''
	global x
	csvPath = Path('SessionData')/x/(x+'.csv')
	colums = ['trail','regressor','c0_theta','c1_theta','c2_theta','c3_theta']
	if(os.path.exists(csvPath)):
		os.remove(csvPath)

	# Build the trail column
	trail = np.arange(0,bandPower.shape[0]).reshape(-1,1)
	
	# Get the regressor output for all trails
	regressorOut = []
	for s in bandPower:
		s = s.mean(axis=0)
		out = regressor(s)
		regressorOut.append(out)
	regressorOut = np.asarray(regressorOut).reshape(-1,1)

	# Get theta band of all the channels for all n trails
	# Select only the theta band
	chanOut = bandPower[:,:,1]
	
	# Put them all together
	data = np.hstack((trail,regressorOut,chanOut))	
	df = pd.DataFrame(data,columns=colums)
	df.to_csv(csvPath,index=False)
	return data

def regressor(out):
	
	outs = np.array([-0.0651,-0.0136,0.0256,-0.0072,0.0009,-0.0032])

	out = np.append(1,out)

	val = np.dot(outs,out)
	
	return abs(val)

newData = preprocessData(data)
band = processData(newData)
_ = saveDataToCSV(band)
