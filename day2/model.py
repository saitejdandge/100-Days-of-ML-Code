from utils import *
from matplotlib import pyplot
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D, Dropout, Activation,MaxPooling2D,Flatten
from sklearn.model_selection import train_test_split
from keras.models import Sequential
import tkinter as tk
from tkinter.filedialog import askdirectory
import numpy as np
from sklearn.metrics import classification_report

#original_data_dir = askdirectory()

dir_name=input("Enter images dir")

original_data_dir  = os.path.dirname(os.path.realpath(__file__))+"/"+str(dir_name)

base_folder=os.path.basename(original_data_dir)

print('Base_Folder ',base_folder)
	
x=get_value('x',base_folder)

y=get_value('y',base_folder)

if x is None or y is None:
	
	print("Please run preprocess.py before running model.py ...")
	
	exit()
	
	pass

print('x shape',x.shape)

print('y shape',y.shape)


x_train,x_val,y_train,y_val=train_test_split(x,y)


x_train=x_train.astype('float32')

x_train/=255


x_val_scaled=x_val.astype('float32')

x_val_scaled/=255


#Building model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=x.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(y.shape[1]))
model.add(Activation('softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

history=model.fit(x_train,y_train,epochs=20,validation_data=(x_val_scaled,y_val), batch_size=10, verbose=1)

if not (os.path.isdir(base_folder+"/model_data")):
	os.makedirs(base_folder+"/model_data")
	pass

model_name="model"

model.save_weights(base_folder+'/model_data/'+model_name+'_weights.h5')

model.save(base_folder+'/model_data/'+model_name+'.h5')

print("Model saved")
#model.evaluate(x_val,y_val)

# pyplot.plot(history.history['acc'],label='Training Accuracy')

# pyplot.plot(history.history['val_acc'],label='Validation Accuracy')

# pyplot.legend()

#print(classification_report(convert_to_labels(y_val,base_folder),convert_to_labels(model.predict(x_val_scaled),base_folder)))

show_prediction_images(x_val,y_val,model,base_folder)

pyplot.show()