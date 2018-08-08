import numpy as np
from utils import *
import emoji
import pandas as pd
from keras.models import load_model

is_emojify=True

def take_input():

	x_vectors=[]

	print("Please enter message...")

	x_vectors.append(sentence_to_avg(input(), word_to_vec_map))

	x_vectors=np.array(x_vectors)


	if is_emojify:

		print(label_to_emoji_list(convert_to_labels(model.predict(x_vectors)))[0])
	
	else:
		print(convert_to_labels(model.predict(x_vectors))[0])
	

	pass



model=load_model('models/softmax.h5')


word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

mapping=get_mapping()

print('Mapping')

print(mapping)

print("Please enter number of inputs...")

itera=int(input())


for x in range(1,itera+1):

	take_input()
	
	pass

