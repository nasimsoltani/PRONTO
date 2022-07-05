import keras
import keras.models as models
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, Conv1D, MaxPooling1D
from keras.layers.core import Flatten, Dense, Dropout, Activation, Reshape
from keras.models import model_from_json#, load_weights
import os

def pronto_l(args):

    if args.cfo_estimation:
        last_layer_activation = 'tanh'
    else:
        last_layer_activation = 'softmax'
   
    model = models.Sequential()
    model.add(Conv1D(128,7, padding='same', input_shape=(args.slice_size, 2)))
    model.add(keras.layers.BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))
    model.add(Conv1D(128,5, padding='same'))
    model.add(keras.layers.BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))
    model.add(MaxPooling1D())
    for i in range(1, 5):
        model.add(Conv1D(128,7, padding='same'))
        model.add(keras.layers.BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(Conv1D(128,5, padding='same'))
        model.add(keras.layers.BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(args.num_classes, activation=last_layer_activation))

    return model

def pronto_s(args):
    
    if args.cfo_estimation:
        last_layer_activation = 'tanh'
    else:
        last_layer_activation = 'softmax'

    model = models.Sequential()
    model.add(Conv1D(64,3, padding='same', input_shape=(args.slice_size, 2)))
    model.add(keras.layers.BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))
    model.add(MaxPooling1D())

    for i in range (0,2):       
        model.add(Conv1D(64,3, padding='same'))
        model.add(keras.layers.BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling1D())

    model.add(Flatten())

    model.add(Dense(128,activation='relu'))
    model.add(Dense(128,activation='relu'))

    model.add(Dense(args.num_classes,activation=last_layer_activation))
    
    return model

def create_model(args):
    """ creates a new model, or loads a model structure and inserts weights 
    and returns either and empty model or a pre-trained model"""


    if args.contin:
        
        # reading model from json file
        json_file = open(args.json_path, 'r')
        model = model_from_json(json_file.read(), custom_objects=None)
        json_file.close()
        model.load_weights(args.hdf5_path, by_name=True)
    
    else:
        if args.model_flag == 'pronto-l':
            model = pronto_l(args)
            print('PRONTO_L model loaded')
        else: #'pronto-s'
            model = pronto_s(args)
            print('PRONTO_S model loaded')
        
        # save the newly created model
        model_json = model.to_json()
        with open(os.path.join(args.save_path,'model_file.json'), "w") as json_file:
            json_file.write(model_json)
    
    return model


