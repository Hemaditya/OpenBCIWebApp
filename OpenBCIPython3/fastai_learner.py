from fastai import *
from fastai.vision import *
import os

def train(path=None):
	'''
		Will train on the data
		keep it in the format path/train/...
	'''
	
	if not os.path.exists(path):
		print("The path is invalid")
		return 
	
	bunch = ImageDataBunch.from_folder(path)			
