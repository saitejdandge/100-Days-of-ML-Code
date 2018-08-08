 # Naive LSTM to learn one-char to one-char mapping
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM, Bidirectional
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.models import load_model
import pandas as pd
from sklearn.model_selection import train_test_split
import math
# 	1 	2 	3	4 	5 	6 	7 	8 	9 	10 	11 	12 	13 	14 	15 	16	 17 	18 	19	 20

def value_function(x):
	return math.cos(x)
	pass


dataset=[]



x_values=list(range(1,50))

for i in x_values:
	dataset.append(value_function(i))
	pass

# df=pd.read_csv('sine-wave.csv')

# dataset=df.values

window_size=3

dataset=numpy.array(dataset)

print('dataset shape ',dataset.shape)

print(x_values)
print(dataset)

dataset=dataset[0:dataset.shape[0]-int(dataset.shape[0]%window_size)]

X=dataset.reshape(int(dataset.shape[0]/window_size),window_size,1)

X=numpy.array(X)

Y=[]

flat_dataset=dataset.flatten()

Y=flat_dataset[window_size:flat_dataset.shape[0]:window_size]

Y=Y.reshape(Y.shape + (1,))

X=X[:-1]

print('X shape ',X.shape)

print('Y shape ',Y.shape)

# X shape (3,window_size,1)
# Y shape (3, n_classes)

# # create and fit the model

X_train,X_val,y_train,y_val=train_test_split(X,Y,shuffle=False)

model = Sequential()

model.add(Bidirectional(LSTM(40), input_shape=(X.shape[1], X.shape[2])))

model.add(Dropout(0.25))

model.add(Activation('relu'))

#number_of_classes y.shape[1]

model.add(Dense(1))

model.add(Activation("linear"))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit(X, Y,validation_data=(X_val,y_val), epochs=2000, batch_size=1, verbose=1)

# summarize performance of the model

print('Model Saved')

model.save('my_model.h5') 


plt.plot(model.predict(X),'r',label='Predictions')

plt.plot(Y,'y',label='Original')


scores = model.evaluate(X_val, y_val, verbose=0)

print("Model Validation Accuracy: %.2f%%" % (scores[1]*100))

plt.legend()

plt.show()

def make_forward_prediction():
	Y_temp=model.predict(X_train[-1])
	pass
