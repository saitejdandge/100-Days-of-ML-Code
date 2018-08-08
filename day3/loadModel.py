 # Naive LSTM to learn one-char to one-char mapping
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM, Bidirectional
from keras.utils import np_utils
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

model = load_model('my_model.h5')

dataset=[]

df=pd.read_csv('sine-wave.csv')

dataset=df.values

dataset=dataset[:1000]

window_size=3

dataset=numpy.array(dataset)

dataset=dataset[0:dataset.shape[0]-int(dataset.shape[0]%window_size)]

X=dataset.reshape(int(dataset.shape[0]/window_size),window_size,1)

X=numpy.array(X)

Y=[]

flat_dataset=dataset.flatten()

Y=flat_dataset[window_size:flat_dataset.shape[0]:window_size]

Y=Y.reshape(Y.shape + (1,))

X=X[:-1]

# print(X.shape)

# print(Y.shape)

# from keras.utils import to_categorical
# y = to_categorical(Y)

# X shape (3,window_size,1)
# Y shape (3, n_classes)

# # create and fit the model

X_train,X_val,y_train,y_val=train_test_split(X,Y)

# print(X_val.shape,"x validation shape")

# print(y_val.shape," y validation shape")

# predictions=model.predict(X_val)

# # print(predictions)

# print("Original")

# print(y_val)


# plt.plot(predictions,'r+',label='predictions')
# plt.plot(y_val,'y+',label='actual')
# plt.plot(predictions,'r')
# plt.plot(y_val,'y')


scores = model.evaluate(X_val, y_val, verbose=0)
#print("Model Validation Accuracy: %.2f%%" % (scores[1]*100))
# plt.legend()

# X shape (3,window_size,1)
# Y shape (3, n_classes)
predictions=[]
def moveForward(steps):

	temp=numpy.array(X_val[-1],ndmin=3)

	print('init ',temp)

	print(temp.shape,' is temp shape')

	for x in range(1,steps):

		print('step ',x)
		y=model.predict(temp)


		temp[0][0][0]=temp[0][1][0]
		temp[0][1][0]=temp[0][2][0]
		temp[0][2][0]=y

		predictions.append(y)
		pass
	pass
moveForward(5000)


plt.plot(numpy.array(predictions).flatten(),'r')
plt.show()