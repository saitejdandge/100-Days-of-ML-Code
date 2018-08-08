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

# 	1 	2 	3	4 	5 	6 	7 	8 	9 	10 	11 	12 	13 	14 	15 	16	 17 	18 	19	 20

dataset=[]

df=pd.read_csv('sine-wave.csv')

dataset=df.values



# for x in range(1,30):
	
# 	dataset.append(x)
	
# 	pass

window_size=3

dataset=numpy.array(dataset)


print('dataset shape ',dataset.shape)
dataset=dataset[0:dataset.shape[0]-int(dataset.shape[0]%window_size)]

X=dataset.reshape(int(dataset.shape[0]/window_size),window_size,1)

X=numpy.array(X)

Y=[]

flat_dataset=dataset.flatten()

Y=flat_dataset[window_size:flat_dataset.shape[0]:window_size]

Y=Y.reshape(Y.shape + (1,))

X=X[:-1]

print(X.shape)

print(Y.shape)

# from keras.utils import to_categorical
# y = to_categorical(Y)

# X shape (3,window_size,1)
# Y shape (3, n_classes)

# # create and fit the model

X_train,X_val,y_train,y_val=train_test_split(X,Y)

print(X_val.shape,"x validation shape")

print(y_val.shape," y validation shape")

model = Sequential()

# model.add(LSTM(40, input_shape=(X.shape[1], X.shape[2])))  #(3,1)

model.add(Bidirectional(LSTM(40), input_shape=(X.shape[1], X.shape[2])))
#model.add(Dropout(0.25))


#number_of_classes y.shape[1]
model.add(Dense(1))

model.add(Activation("linear"))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20000, batch_size=1, verbose=2)
# summarize performance of the model
model.save('my_model.h5') 
predictions=model.predict(X_val)

print(predictions)

print("Original")

print(y_val)


plt.plot(predictions,'r+',label='predictions')
plt.plot(y_val,'y+',label='actual')
plt.plot(predictions,'r')
plt.plot(y_val,'y')


scores = model.evaluate(X_val, y_val, verbose=0)
print("Model Validation Accuracy: %.2f%%" % (scores[1]*100))
plt.legend()
plt.show()

def make_forward_prediction():
	Y_temp=model.predict(X_train[-1])
	pass


# demonstrate some model predictions
# for pattern in dataX:
# 	x = numpy.reshape(pattern, (1, len(pattern), 1))
# 	x = x / float(len(alphabet))
# 	prediction = model.predict(x, verbose=0)
# 	index = numpy.argmax(prediction)
# 	result = int_to_char[index]
# 	seq_in = [int_to_char[value] for value in pattern]
# 	print (seq_in, "->", result)





