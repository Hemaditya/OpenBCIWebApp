import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import simps

def mean(d,ax=-1):
	# The input shape to this funtion should be (channels,batch_size,chunk_size) -> (8,100,250)
	return np.mean(d,axis=ax)

def variance(d,ax=-1):
	# The input shape to this funtion should be (channels,batch_size,chunk_size) -> (8,100,250)
	return np.var(d,axis=ax)

def skewness(d):
	# The input shape to this funtion should be (channels,batch_size,chunk_size) -> (8,100,250)
	# s = a/b
	# a = (1/N)*sum((sample[i]-mean)^3)
	# b = ((1/N-1)*sum(sample[i]-mean)^2)^3/2
	if(len(d.shape) != 3):
		print("The shape of input data must be (channels,samples,chunk_size)")
		return

	a = float((1/d.shape[1]))
	mu = np.expand_dims(mean(d),-1)
	Pow = np.power(d-mu,3)
	Sum = np.sum(Pow,axis=-1)
	a = a*Sum

	b = float(1/(d.shape[1]-1))
	Pow = np.power(d-mu,2)
	Sum = np.sum(Pow,axis=-1)
	b = np.power(b*Sum,3)
	b = np.power(b,1/2)
	
	skew = a/b
	return skew

def kurtosis(d):
	# The input shape to this funtion should be (channels,batch_size,chunk_size) -> (8,100,250)
	# s = a/b
	# a = (1/N)*sum((sample[i]-mean)^4)
	# b = ((1/N)*sum((sample[i]-mean)^2))^2

	if(len(d.shape) != 3):
		print("The shape of input data must be (channels,samples,chunk_size)")
		return
	a = float(1/d.shape[1])
	mu = np.expand_dims(mean(d),-1)
	Pow = np.power(d-mu,4)
	Sum = np.sum(Pow,axis=-1)
	a = a*Sum

	b = float(1/(d.shape[1]-1))
	Pow = np.power(d-mu,2)
	Sum = np.sum(Pow,axis=-1)
	b = np.power(b*Sum,2)

	kurtosis = a/b - 3.0
	return kurtosis

def zero_crossing(d):
	# The input shape to this funtion should be (channels,batch_size,chunk_size) -> (8,100,250)
	# z = sum(sample[i]*sample[i-1])
	if(len(d.shape) != 3):
		print("The shape of input data must be (channels,samples,chunk_size)")
		return
	
	d_roll = np.roll(d,shift=-1,axis=-1)
	d = d*d_roll
	d = d[...,0:-1]
	zero_cross = np.sum(d,axis=-1)

	return zero_cross

def area_under_signal(d):
	# The input shape to this funtion should be (channels,batch_size,chunk_size) -> (8,100,250)
	if(len(d.shape) != 3):
		print("The shape of input data must be (channels,samples,chunk_size)")
		return
	out = simps(d,dx=250.0)	
	return out

def peak2peak(d):
	# The input shape to this funtion should be (channels,batch_size,chunk_size) -> (8,100,250)
	# max(sample) - min(sample)
	if(len(d.shape) != 3):
		print("The shape of input data must be (channels,samples,chunk_size)")
		return

	_max = np.max(d,axis=-1)
	_min = np.min(d,axis=-1)
	
	p2p = _max - _min
	return p2p

def band_power(d):
	# The input shape to this funtion should be (channels,batch_size,chunk_size) -> (8,100,250)
	if(len(d.shape) != 3):
		print("The shape of input data must be (channels,samples,chunk_size)")
		return
	delta = (0.5,4)
	theta = (4,8)
	alpha = (8,12)
	beta = (12,30)
	
	freqs, psd = signal.welch(d,250,nperseg=500)
	idx_delta = np.where(np.logical_and(freqs>=delta[0],freqs<=delta[1]))
	idx_theta = np.where(np.logical_and(freqs>=theta[0],freqs<=theta[1]))
	idx_alpha = np.where(np.logical_and(freqs>=alpha[0],freqs<=alpha[1]))
	idx_beta = np.where(np.logical_and(freqs>=beta[0],freqs<=beta[1]))
	
	delta_band_pow = simps(psd[...,idx_delta],dx=(delta[0]-delta[1]))
	theta_band_pow = simps(psd[...,idx_theta],dx=(theta[0]-theta[1]))
	alpha_band_pow = simps(psd[...,idx_alpha],dx=(alpha[0]-alpha[1]))
	beta_band_pow = simps(psd[...,idx_beta],dx=(beta[0]-beta[1]))
	
	bandPower = (delta_band_pow,theta_band_pow,alpha_band_pow,beta_band_pow)

	return bandPower

def test():
	# This function is only for test purposes and debugging
	x = np.random.randint(0,5,size=(3,100,250))
	a = mean(x)
	b = variance(x)
	c = skewness(x)
	d = kurtosis(x)
	e = zero_crossing(x)
	f = area_under_signal(x)
	g = peak2peak(x)

	h = band_power(x)

	print(a.shape)
	print(b.shape)
	print(c.shape)
	print(d.shape)
	print(e.shape)
	print(f.shape)
	print(g.shape)
	print(h[0].shape)
	print(h[1].shape)
	print(h[2].shape)
	print(h[3].shape)

