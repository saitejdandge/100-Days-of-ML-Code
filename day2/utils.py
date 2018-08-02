from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import pickle

def convert_to_one_hot(Y):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(Y)
    le_name_mapping = dict(zip(label_encoder.transform(label_encoder.classes_),label_encoder.classes_))
    set_value('mapping',le_name_mapping)
    return label_encoder, to_categorical(integer_encoded)

def get_value(key):
    
    pickle_in = open("data/data.pickle","rb")
    example_dict = pickle.load(pickle_in)
    pickle_in.close()
    return example_dict[key]
    
    pass

def set_value(key,value):
    pickle_in = open("data/data.pickle","rb")
    example_dict = pickle.load(pickle_in)
    example_dict[key] = value
    pickle_out = open("data/data.pickle","wb")
    pickle.dump(example_dict, pickle_out)
    pickle_out.close()
    pickle_in.close()
    pass