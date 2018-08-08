import numpy as np
from utils import *
import emoji
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Embedding
from keras.layers import LSTM, Bidirectional
from keras.models import load_model, save_model
from matplotlib import pyplot
from keras.models import Model

dataFrame=pd.read_csv('data/train.tsv',sep='\t')

x=dataFrame.values[:,2]

y=dataFrame.values[:,3]

#maxLen = len(max(x, key=len).split())

maxLen = 20

print(maxLen, " is each sentence max length")

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

label_encoder,encoded=convert_to_one_hot(y)

# x_vectors=[]

# for i in x:
# 	x_vectors.append(sentence_to_avg(i, word_to_vec_map))
# 	pass

# x_vectors=np.array(x_vectors)

#Building model
# X1 = np.array(["funny lol", "lets play baseball", "food is ready for you"])
# X1_indices = sentences_to_indices(X1,word_to_index, max_len = 5)
# print("X1 =", X1)
# print("X1_indices =", X1_indices)

# embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
# print("weights[0][1][3] =", embedding_layer.get_weights()[0][1][3])

# def Emojify_V2(input_shape, word_to_vec_map, word_to_index):

#     sentence_indices = Input(input_shape, dtype='int32')

#         # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
#     embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    
#     # Propagate sentence_indices through your embedding layer, you get back the embeddings
#     embeddings = embedding_layer(sentence_indices)   
    
#     # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
#     # Be careful, the returned output should be a batch of sequences.

#     print(embeddings)

#     X = LSTM(128, return_sequences=True)(embeddings)
#     # Add dropout with a probability of 0.5
#     X = Dropout(0.5)(X)
#     # Propagate X trough another LSTM layer with 128-dimensional hidden state
#     # Be careful, the returned output should be a single hidden state, not a batch of sequences.
#     X = LSTM(128, return_sequences=False)(X)
#     # Add dropout with a probability of 0.5
#     X = Dropout(0.5)(X)
#     # Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
#     X = Dense(5)(X)
#     # Add a softmax activation
#     X = Activation('softmax')(X)
    
#     # Create Model instance which converts sentence_indices into X.
#     model = Model(inputs=sentence_indices, outputs=X)
    
#     ### END CODE HERE ###
    
#     return model


# model = Emojify_V2((maxLen,), word_to_vec_map, word_to_index)


# print(model.summary())

X_train_indices = sentences_to_indices(x, word_to_index, maxLen)

x_train,x_val,y_train,y_val=train_test_split(X_train_indices,encoded)

model=Sequential()

embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index,False)

# sentence_indices = Input((maxLen,), dtype='int32')

# print('embedding layer embeddings',embedding_layer(sentence_indices))

model.add(embedding_layer)

model.add(LSTM(128,return_sequences=True))

model.add(Dropout(0.5))

model.add(LSTM(128, return_sequences=False))

model.add(Dropout(0.5))

model.add(Dense(5,activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

#model.load_weights('models/lstm_weights.h5')

history = model.fit(x_train, y_train, epochs=100, batch_size=200, verbose=1,validation_data=(x_val,y_val))

pyplot.plot(history.history['acc'],label='Training Accuracy')

pyplot.plot(history.history['val_acc'],label='Validation Accuracy')

model.save('models/lstm_sentiment_kaggle.h5')

model.save_weights('models/lstm_sentiment_kaggle_weights.h5')

print("model saved")

print(model.summary())

pyplot.legend()

pyplot.show()
