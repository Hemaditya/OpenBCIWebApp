import matplotlib.pyplot as plt
import numpy as np

def plotter(data,c=range(8)):
	'''
		A playback function for recorded data
		Input:
			- $data of shape (n_chunks, channels, chunk_size)
			- $c is the channels that need to be plot, 
			  Ex: if c is a single number then it will plot only that channel
				  if c is a list then it will plot those channels
		Output:
			- Plots the data
	'''
	data = data.transpose(1,0,2)
	if(type(c) == int):
		if(c > 7 and c < 0):
			print("Invalid Channel given, Expected c to be in range of 0 to 8")
			return 0

		channel_data = data[c]	
		plt.plot(channel_data.reshape(-1))

	elif(type(c) == range or type(c) == list):
		c = np.array(c)

		if((c < 0).any() or (c > 7).any() or c.shape[0] > 8):
			print("The channels given in c are not in the range of 0 to 8")
			return 0

		channel_data = data[c]
		n_channels = data.shape[0]

		fig,ax = plt.subplots(
