import matplotlib.pyplot as plt
import os

def savefig(data,path=None):
	'''	
		This function expects data in the shape of (n_chunks,channels,freq,chunk_size)
	'''
	if not os.path.exists(path):
		print("Invalid path")
		return

	
	fig, ax = plt.subplots(4,2)
	fig.subplots_adjust(0,0,1,1,0,1)
	ax = ax.reshape(-1)

	for i,s in enumerate(data):
		print(i,"/",data.shape[0])
		for c in range(8):
			ax[c].cla()
			ax[c].imshow(s[c],cmap='hot')
		fig.savefig(path+"/data_"+str(i))
	
	plt.show()
