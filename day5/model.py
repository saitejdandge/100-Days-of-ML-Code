import numpy as np
from utils import *
import emoji
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM, Bidirectional
from keras.models import load_model, save_model


dataFrame=pd.read_csv('data/train_emoji.csv')

x=dataFrame.values[:,0]
y=dataFrame.values[:,1]

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

label_encoder, mapping, encoded=convert_to_one_hot(y)


def sentence_to_avg(sentence, word_to_vec_map):
    """
    Converts a sentence (string) into a list of words (strings). Extracts the GloVe representation of each word
    and averages its value into a single vector encoding the meaning of the sentence.
    
    Arguments:
    sentence -- string, one training example from X
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    
    Returns:
    avg -- average vector encoding information about the sentence, numpy-array of shape (50,)
    """
    
    ### START CODE HERE ###
    # Step 1: Split sentence into list of lower case words (≈ 1 line)
    words = sentence.split()
    
    words=[x.lower() for x in words]

    words=np.array(words)
    
    word_vectors=[]
    
    for i in words :
        word_vectors.append(word_to_vec_map[i])
    # Initialize the average word vector, should have the same shape as your word vectors.
    
    word_vectors=np.array(word_vectors)
    
    avg = np.zeros(word_vectors.shape[1])
    
    # Step 2: average the word vectors. You can loop over the words in the list "words".
    for w in word_vectors:
        avg += w
    avg = avg/len(words)
    
    ### END CODE HERE ###
    
    return avg

x_vectors=[]

for i in x:
	x_vectors.append(sentence_to_avg(i, word_to_vec_map))
	pass

x_vectors=np.array(x_vectors)

x_vectors=x_vectors.reshape(x_vectors.shape[0],x_vectors.shape[1],1)

#encoded=encoded.reshape(encoded.shape[0],encoded.shape[1],1)

x_train,x_test,y_train,y_test=train_test_split(x_vectors,encoded)


print('x train shape')
print(x_train.shape)

print('y train shape')
print(y_train.shape)

#Building model

model = Sequential()

model.add(Bidirectional(LSTM(x_train.shape[2]), input_shape=(x_train.shape[1], x_train.shape[2])))

model.add(Dense(y_train.shape[1],activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=70, batch_size=1, verbose=2)

model.save_model('my_model.h5')
#model=load_model('my_model.h5') 

# predictions=model.predict(x_test)

# print(predictions)


