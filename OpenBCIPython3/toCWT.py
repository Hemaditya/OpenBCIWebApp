import pywt
import numpy as np

def toCWT(data):
	'''
		Converts data to CWT
		$data Should be in format (n_chunks, channels, chunk_size)
		the return k_ shape will be (n_chunks,channels,frquency,chunk_size)
	'''
	f = pywt.scale2frequency('gaus1',np.arange(1,129))*250.0
	f = np.logical_and(f>=1,f<=50)
	scales = np.arange(1,129)[f]
	k_ = []
	for i,s in enumerate(data):
		print("Processed Sample: ",i)
		m_ = []
		for j in range(data.shape[1]):
			m_.append(pywt.cwt(s[j],scales,'gaus1')[0])
		k_.append(m_)
	return np.asarray(k_)
