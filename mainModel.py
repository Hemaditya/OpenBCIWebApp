import poo
from sklearn.preprocessing import normalize
import numpy as np
import utils
import LSTM
import kerasModel

# To be able to use data outside
d = []

def preprocess_data():
	global d
	data = np.copy(poo.daata)
	# Drop the first two samples beacause they are just zero
	data = data[2:]
	# Reshape the data to (channels,samples,chunk_size)
	data = np.transpose(data,(1,0,2))			
	a = np.expand_dims(utils.mean(data),-1) # (8,180)
	b = np.expand_dims(utils.variance(data),-1) #(8,180)
	c = np.expand_dims(utils.skewness(data),-1) #(8,180)
	d = np.expand_dims(utils.kurtosis(data),-1) #(8,180)
	e = np.expand_dims(utils.zero_crossing(data),-1) #(8,180)
	f = np.expand_dims(utils.area_under_signal(data),-1) #(8,180)
	g = np.expand_dims(utils.peak2peak(data),-1) #(8,180)
	h = utils.band_power(data) #(8,180,1)
	
	# Stack all the 11 features together
	finalData = a
	finalData = np.append(finalData,b,-1)
	finalData = np.append(finalData,c,-1)
	finalData = np.append(finalData,d,-1)
	finalData = np.append(finalData,e,-1)
	finalData = np.append(finalData,f,-1)
	finalData = np.append(finalData,g,-1)
	finalData = np.append(finalData,h[0],-1)
	finalData = np.append(finalData,h[1],-1)
	finalData = np.append(finalData,h[2],-1)
	finalData = np.append(finalData,h[3],-1)

	ranges_a = list(range(0,12,3))
	ranges_b = list(range(1,12,3))
	ranges_c = list(range(2,12,3))

	idx_a = np.asarray([np.arange(i*15,(i+1)*15) for i in ranges_a]).reshape(-1)
	idx_b = np.asarray([np.arange(i*15,(i+1)*15) for i in ranges_b]).reshape(-1)
	idx_c = np.asarray([np.arange(i*15,(i+1)*15) for i in ranges_c]).reshape(-1)
	
	zeros = np.zeros(shape=(8,180,1))

	zeros[:,idx_a,-1] = 0
	zeros[:,idx_b,-1] = 1
	zeros[:,idx_c,-1] = 2

	finalData = np.copy(poo.daata)
	finalData = finalData[2:,:,0:-1]
	finalData = np.transpose(finalData,(1,0,2))
	print(finalData.shape)
	#finalData[finalData >= 400.0] = 400.0

	#finalData[finalData <= -400.0] = -400.0

	for i, sample in enumerate(finalData):
		finalData[i] = normalize(sample)

	finalData = np.append(finalData,zeros,-1)
	d = np.copy(finalData)

	kerasModel.modelTrain(finalData[[3,4,6],:,:])

def test():
	preprocess_data()

test()
