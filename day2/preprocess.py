import matplotlib.image as mpimg
import os
from PIL import Image
import numpy as np
import PIL
import matplotlib.pyplot as plt
from utils import *
import os
import tkinter as tk
from tkinter.filedialog import askdirectory

dir_path = os.path.dirname(os.path.realpath(__file__))

#original_data_dir = askdirectory()
dir_name=input("Enter images dir")

original_data_dir  = os.path.dirname(os.path.realpath(__file__))+"/"+str(dir_name)

base_folder=os.path.basename(original_data_dir)

print('Base_Folder ',base_folder)

img_height=img_width=300

squared_dir=original_data_dir+"/preprocessed_images"

print("Rescaling Images to dimension ",str(img_width)+" * "+str(img_height))

for subdir, dirs, files in os.walk(original_data_dir+"/original"):
   
    for file in files:
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file

        if filepath.endswith(".jpg"):

        	class_name=os.path.basename(os.path.dirname(filepath))

        	new_file_path=str(squared_dir)+"/"+str(class_name)+"/"+str(file)

        	square_image(filepath,new_file_path,img_width,img_height)
   

print("Finished Rescaling Images")

x=[]
y=[]

print("Converting Images to Arrays")

for subdir, dirs, files in os.walk(original_data_dir+"/preprocessed_images/"):
   
    for file in files:
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file

        if filepath.endswith(".jpg"):

            class_name=os.path.basename(os.path.dirname(filepath))

            new_file_path=str(squared_dir)+"/"+str(class_name)+"/"+str(file)

            read_image(filepath,class_name,x,y)

x=np.array(x)

y=np.array(y)

print("One-Hot Encoding Labels")

le,y=convert_to_one_hot(y,base_folder)



print("Saving X, Y objects locally")

set_value('x',x,base_folder)

set_value('y',y,base_folder)

print("Finished X, Y objects locally")

print('mapping ',get_value('mapping',base_folder))

print('x shape',get_value('x',base_folder).shape)

print('y shape',get_value('y',base_folder).shape)

print("Finished Preprocessing, Run model.py and select folder.")






