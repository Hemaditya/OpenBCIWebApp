import numpy as np

def distance(a,b):
	a = np.array(a)
	b = np.array(b)
	return np.linalg.norm(a-b)

def timeEntropy(N,m,r,arr):
	component = []
	for i in range(N-m+1):
		d = arr[i:i+m]
		sim = []
		for j in range(N-m+1):
			if(j != i):	
				if(np.sqrt(distance(d,arr[j:j+m]))<=r):
					sim.append(1)
				else:
					sim.append(0)
			else:
				pass
		ci = sum(sim)/float(len(sim))
		component.append(ci)
	
	amr = (1.0/float(N-m+1)) * sum([val for val in component])
	return amr

data = np.random.randint(0,15,size=(250))
