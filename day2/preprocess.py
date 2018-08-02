import matplotlib.image as mpimg
import os
from PIL import Image
import numpy as np
import PIL
import matplotlib.pyplot as plt
from utils import *
import os

img_width=300

img_height=300

dir_path = os.path.dirname(os.path.realpath(__file__))

# # Print out its shape
# print(image.shape)
# plt.imshow(image)
# plt.show()


def square_image(filename,path):
	im = Image.open(filename)
	#sqrWidth = np.ceil(np.sqrt(im.size[0]*im.size[1])).astype(int)
	im_resize = im.resize((img_width, img_height))
	if not os.path.exists(os.path.dirname(path)):
  	  os.makedirs(os.path.dirname(path))
	im_resize.save(path)
	pass

squared_dir=dir_path+"/squared_images"

print("Rescaling Images to dimension ",str(img_width)+" * "+str(img_height))

for subdir, dirs, files in os.walk(dir_path+"/flowers/"):
   
    for file in files:
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file

        if filepath.endswith(".jpg"):

        	class_name=os.path.basename(os.path.dirname(filepath))

        	new_file_path=str(squared_dir)+"/"+str(class_name)+"/"+str(file)

        	square_image(filepath,new_file_path)
   
print("Finished Rescaling Images")

x=[]
y=[]

print("Converting Images to Arrays")

# Reading image
def read_image(filepath,class_name):
    image = mpimg.imread(filepath)
    x.append(image)
    y.append(class_name)
    pass

for subdir, dirs, files in os.walk(dir_path+"/squared_images/"):
   
    for file in files:
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file

        if filepath.endswith(".jpg"):

            class_name=os.path.basename(os.path.dirname(filepath))

            new_file_path=str(squared_dir)+"/"+str(class_name)+"/"+str(file)

            read_image(filepath,class_name)

x=np.array(x)

y=np.array(y)

le,y=convert_to_one_hot(y)

print("Finished Converting Images to Arrays")

print("Saving X, Y objects locally")

set_value('x',x)

set_value('y',y)

print("Finished X, Y objects locally")

print('mapping ',get_value('mapping'))

print('x shape',get_value('x').shape)

print('y shape',get_value('y').shape)








