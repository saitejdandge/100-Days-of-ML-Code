
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from utils import *
import matplotlib.pyplot as plt

print("Loading Word Embeddings....")
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('glove.6B.50d.txt')

words=[]
vectors=[]

for i in word_to_vec_map.keys():
	words.append(i)
	vectors.append(word_to_vec_map[i])
	pass

print('Saving Word Embeddings to TSV File...')
pd.DataFrame(data=vectors[:10000]).to_csv('data.tsv',sep='\t',header=None,index=False)
pd.DataFrame(data=words[:10000]).to_csv('metadata.tsv',sep='\t',header=None,index=False)

print('Saved to data.tsv and metadata.tsv ...')
