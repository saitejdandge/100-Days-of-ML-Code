from utils import *
from matplotlib import pyplot
from keras.layers import Dense, Conv2D, Dropout, Activation,MaxPooling2D,Flatten
from sklearn.model_selection import train_test_split
from keras.models import Sequential

x=get_value('x')
y=get_value('y')


x = x.astype('float32')
x /= 255

print('x shape',x.shape)
print('y shape',y.shape)


x_train,x_val,y_train,y_val=train_test_split(x,y)

#Building model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=x.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(y.shape[1]))
model.add(Activation('softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

history=model.fit(x_train,y_train,epochs=20,validation_data=(x_val,y_val), batch_size=10, verbose=1)

if not (os.path.isdir("/model_data")):
	os.makedirs("model_data")
	pass

model.save_weights('model_data/first_try.h5')

model.evaluate(x_val,y_val)

pyplot.plot(history.history['acc'],label='Training Accuracy')

pyplot.plot(history.history['val_acc'],label='Validation Accuracy')

pyplot.legend()

pyplot.show()