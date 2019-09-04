import mneProgram as MNE
import numpy as np
import time

files = MNE.retrieveFiles()
files = files["pickle"]
acc1 = 0
band =  [0,0]

for i in range(30,12,-2):
	data,labels = MNE.dataFromFiles(files)
	indices = list(range(3,data.shape[0],6))
	indices = np.asarray([[i,i+1,i+2] for i in indices]).reshape(-1)
	#data = data[indices]
	#print(data.shape,labels.shape)
	mneobject = MNE.rawArrayToObject(data)
	#epochs = MNE.buildepochs(mneobject,labels,policy=1)
	mneobject.notch_filter([50.0])
	mneobject.filter(8,i)
	epochs = MNE.buildEpochs(mneobject,labels)
	acc = MNE.cspSVMPipeline(epochs)
	if(acc > acc1):
		acc1 = acc
		band[0] = i
		band[1] = 8
	time.sleep(1)
	#MNE.densemodel(epochs,channel=[2,4])
	#MNE.plotchannel(epochs)

print("Best accurac range: ",band)

for i in range(8,band[0],-2):
	data,labels = MNE.dataFromFiles(files)
	indices = list(range(3,data.shape[0],6))
	indices = np.asarray([[i,i+1,i+2] for i in indices]).reshape(-1)
	#data = data[indices]
	#print(data.shape,labels.shape)
	mneobject = MNE.rawArrayToObject(data)
	#epochs = MNE.buildepochs(mneobject,labels,policy=1)
	mneobject.notch_filter([50.0])
	mneobject.filter(i,30)
	epochs = MNE.buildEpochs(mneobject,labels)
	acc = MNE.cspSVMPipeline(epochs)
	if(acc > acc1):
		acc1 = acc
		band[0] = i
		band[1] = 8
	time.sleep(1)
	#MNE.densemodel(epochs,channel=[2,4])
	#MNE.plotchannel(epochs)
print("Best accurac range: ",band)
