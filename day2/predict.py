
from keras.models import load_model
from utils import *
import matplotlib.image as mpimg
import sys

dir_name=input("Enter images dir")

original_data_dir  = os.path.dirname(os.path.realpath(__file__))+"/"+str(dir_name)

base_folder=os.path.basename(original_data_dir)

print(original_data_dir+'/model_data/model.h5')

model=load_model(original_data_dir+'/model_data/model.h5')

filename=sys.argv[1]

squared_image=square_image(filename,300,300)

x=mpimg.imread(squared_image)

x=x[:,:,:3] #removing alpha channel

x=x.reshape([1,x.shape[0],x.shape[1],x.shape[2]])

print(convert_to_labels(model.predict(x),base_folder)[0])