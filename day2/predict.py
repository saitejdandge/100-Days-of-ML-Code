import tkinter as tk
from tkinter.filedialog import askopenfilename
from keras.models import load_model
from utils import *
# tk().withdraw() # we don't want a full GUI, so keep the root window from appearing

model=load_model('model_data/flower_classifier.h5')

filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
print(filename)