from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import matplotlib.image as mpimg
from PIL import Image
from numpy import argmax
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlopen

def convert_to_one_hot(Y,base_folder):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(Y)
    le_name_mapping = dict(zip(label_encoder.transform(label_encoder.classes_),label_encoder.classes_))
    set_value('mapping',le_name_mapping,base_folder)
    return label_encoder, to_categorical(integer_encoded)



# Reading image

def read_image(filepath,class_name,x,y):

    image = mpimg.imread(filepath)

    if image is not None:
        image=image[:,:,:3] #removing alpha channel
        x.append(image)
        y.append(class_name)

    pass

def get_value(key,base_folder):
    if os.path.exists(base_folder+"/pickle_objects/data.pickle"):

        pickle_in = open(base_folder+"/pickle_objects/data.pickle","rb")

        example_dict = pickle.load(pickle_in)

        pickle_in.close()

        return example_dict[key]
    else:
        return None
    pass

def convert_to_labels(Y,base_folder):

    arr=[]
    for x in Y:
        arr.append(get_value('mapping',base_folder)[argmax(x)])
        pass

    arr=np.array(arr)
    return arr

def square_image(path,img_width,img_height):


    im = Image.open(path)
    
    #sqrWidth = np.ceil(np.sqrt(im.size[0]*im.size[1])).astype(int)
    
    im_resize = im.resize((img_width, img_height))
    
    if not os.path.exists(os.path.dirname(path)):
      os.makedirs(os.path.dirname(path))
    
    im_resize.save(path)

    return path

def set_value(key,value,base_folder):
    example_dict={}
    if os.path.exists(base_folder+"/pickle_objects/data.pickle"):
        pickle_in = open(base_folder+"/pickle_objects/data.pickle","rb")
        example_dict = pickle.load(pickle_in)
        pickle_in.close()
        pass
    example_dict[key] = value

    if not os.path.exists(base_folder+"/pickle_objects"):
        os.makedirs(base_folder+"/pickle_objects")
        pass
    pickle_out = open(base_folder+"/pickle_objects/data.pickle","wb")
    pickle.dump(example_dict, pickle_out)
    pickle_out.close()
    
    pass

def show_prediction_images(x_val,y_val,model,base_folder):

    x_val=x_val[:12]
    y_val=y_val[:12]

    fig=plt.figure(figsize=(8, 8))

    columns = 4

    rows = 3

    predicted_labels=convert_to_labels(model.predict(x_val),base_folder)
    original_labels=convert_to_labels(y_val,base_folder)

    for i in range(0, columns*rows ):
       # img = np.random.randint(10, size=(10,10))
        ax=fig.add_subplot(rows, columns, i+1)
        ax.set_title('Original: '+str(original_labels[i])+' Predicted: '+str(predicted_labels[i]))
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.imshow(x_val[i])

    plt.show()
    pass