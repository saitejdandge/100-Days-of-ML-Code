import csv
import numpy as np
import emoji
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
import pickle
from keras.layers import  Embedding,Input
# GRADED FUNCTION: pretrained_embedding_layer
def pretrained_embedding_layer(word_to_vec_map, word_to_index,is_trainable):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.
    
    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """
    
    vocab_len = len(word_to_index) + 1                  # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map["cucumber"].shape[0]      # define dimensionality of your GloVe word vectors (= 50)
    
    ### START CODE HERE ###
    # Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros((vocab_len, emb_dim))
    
    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    print('----------------------------------------------')
    print('----------------------------------------------')
    print("vocabulary length : ", str(vocab_len))
    print("Embedding Dimension : ", str(emb_dim))
    print("Embedding matrix shape : ",emb_matrix.shape)
    print('----------------------------------------------')
    print('----------------------------------------------')
    # Define Keras embedding layer with the correct output/input sizes, make it trainable. Use Embedding(...). Make sure to set trainable=False. 
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=is_trainable)
    ### END CODE HERE ###

    # Build the embedding layer, it is required before setting the weights of the embedding layer. Do not modify the "None".
    embedding_layer.build((None,))
    
    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer

def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4). 
    
    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
    
    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    
    m = X.shape[0]                                   # number of training examples
    
    ### START CODE HERE ###
    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = np.zeros((m, max_len))
    
    not_founds=[]
    for i in range(m):                               # loop over training examples
        
        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        sentence_words = [w.lower() for w in X[i].split()]
        
        # Initialize j to 0
        j = 0
        
        # Loop over the words of sentence_words
        
        

        for w in sentence_words:

            if(j==max_len) :
                break
        
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            if w in word_to_index.keys():
                
                X_indices[i, j] = word_to_index[w]
                pass
            else:
                
                not_founds.append(w)
                X_indices[i, j] = 0
                pass

            
            # Increment j to j + 1
            j += 1
            
    ### END CODE HERE ###
    print('----------------------------------------------')
    print('----------------------------------------------')
    print('X indices shape ',X_indices.shape)
    print(len(not_founds),' words not found in vocabulary')

    
    
    return X_indices


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
    
    keys=word_to_vec_map.keys()
    for i in words :

        if i in keys:
            word_vectors.append(word_to_vec_map[i])
            pass

        else:
            word_vectors.append(np.zeros(word_to_vec_map['hello'].shape))
            pass
        
    # Initialize the average word vector, should have the same shape as your word vectors.
    
    word_vectors=np.array(word_vectors)
    
    avg = np.zeros(word_vectors.shape[1])
    
    # Step 2: average the word vectors. You can loop over the words in the list "words".
    for w in word_vectors:
        avg += w
    avg = avg/len(words)
    
    ### END CODE HERE ###
    
    return avg

def read_glove_vecs(glove_file):
    with open(glove_file, 'r',encoding="utf8") as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map


def convert_to_one_hot(Y):

    label_encoder = LabelEncoder()

    integer_encoded = label_encoder.fit_transform(Y)

    le_name_mapping = dict(zip(label_encoder.transform(label_encoder.classes_),label_encoder.classes_))

    dict_obj = {"mapping":le_name_mapping}
    
    pickle_out = open("data/dict.pickle","wb")
    
    pickle.dump(dict_obj, pickle_out)
    
    pickle_out.close()

    return label_encoder, to_categorical(integer_encoded)


def get_mapping():
    pickle_in = open("data/dict.pickle","rb")
    example_dict = pickle.load(pickle_in)
    return example_dict['mapping']
    pass

def convert_to_labels(Y):

    arr=[]
    for x in Y:
        arr.append(get_mapping()[argmax(x)])
        pass

    arr=np.array(arr)
    return arr


emoji_dictionary = {"0": "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font
                    "1": ":baseball:",
                    "2": ":smile:",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:"}

def label_to_emoji(label):
    """
    Converts a label (int or string) into the corresponding emoji code (string) ready to be printed
    """
    return emoji.emojize(emoji_dictionary[str(label)], use_aliases=True)
    
def label_to_emoji_list(Y):
    """
    Converts a label (int or string) into the corresponding emoji code (string) ready to be printed
    """
    arr=[]
    for x in Y:
        arr.append(label_to_emoji(x))
        pass

    return np.array(arr)   
    
def print_predictions(X, pred):
    print()
    for i in range(X.shape[0]):
        print(X[i], label_to_emoji(int(pred[i])))
        
        
def plot_confusion_matrix(y_actu, y_pred, title='Confusion matrix', cmap=plt.cm.gray_r):
    
    df_confusion = pd.crosstab(y_actu, y_pred.reshape(y_pred.shape[0],), rownames=['Actual'], colnames=['Predicted'], margins=True)
    
    df_conf_norm = df_confusion / df_confusion.sum(axis=1)
    
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    
    