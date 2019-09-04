from app import app
from flask import render_template
#from bciapp import BCI
import random
from modules import properApp
import threading
t = threading.Thread(target=properApp.run)
flag = 0

@app.route('/')

@app.route('/index')
def index():
	user = {'username':'Raghavendr G'}	
	return render_template('index.html',user=user,title='Home')

@app.route('/train')
def train():
	return "Under Construction!"

@app.route('/test')
def test():
	global flag
	if(flag == 0):
		t.start()
		flag = 1
	x = properApp.what_state
	k = properApp.jaa_state
	return render_template("EyeAction.html",x=x,k=k)
