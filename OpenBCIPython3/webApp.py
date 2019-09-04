from flask import Flask, render_template, request
import pickle
import numpy as np
import time
from threading import Thread
import json
import app as A
app = Flask(__name__)
x = input("Enter name: ")
x = x.lower()
if(x == "admin"):
	x = None

chunk_size = 250
bytes_per_sec = 250
bci = A.OpenBCI(path=x,chunk_size=chunk_size)
trials = int(input("Enter the number of trials: "))
i = 0
data = []
def read_data():
	global i,data
	if(i == 0):
		i = i+1
	else:
		time.sleep(0.700)

	data = bci.read_chunk(n_chunks=6*trials*int(bytes_per_sec/chunk_size),save_data=False)
	print("Data reading is finished")

def count():
	for i in range(12):
		print("This is a post request: ",i)
		time.sleep(1)
t = Thread(target=count)

@app.route('/',methods=['GET','POST'])
def hello_world():
	global data
	if request.method == "POST":
		labels = np.asarray(request.form['secret'].split(','))
		labels[labels == 'la'] = 0
		labels[labels == 'ra'] = 1
		labels[labels == 'll'] = 2
		labels[labels == 'rl'] = 3
		t = time.strftime("%Y%m%d_%H%M%S")
		print(labels)
		while(len(data) == 0):
			pass
		pickleData = {'raw':data[0],'labels':labels}
		print("Saving pickle data")
		with open(data[1]+"/"+t+'.pickle','wb') as f:
			pickle.dump(pickleData,f)
			
	return render_template('startup.html')

def getLabels(classes=4):
	
	labels = np.arange(classes).reshape(1,-1)
	labels = np.repeat(labels,4)


@app.route('/trial',methods=['GET','POST'])
def trial():
	t = Thread(target=read_data)
	t.start()
	data = 2
	return render_template('random.html',data = json.dumps(data),trials=trials)
if __name__ == '__main__':
	app.run()

