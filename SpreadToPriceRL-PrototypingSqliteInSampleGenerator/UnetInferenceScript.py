import os
import random

import numpy as np

import tensorflow as tf

import keras.backend
import keras.callbacks
import keras.layers
import keras.models
import keras.optimizers

import cv2
import sys
import keras.layers
import keras.models
import tensorflow as tf

from keras.models import Input, Model
from keras.layers import Conv2D, Concatenate, MaxPooling2D
from keras.layers import UpSampling2D, Dropout, BatchNormalization

from keras.layers import Conv2D, Conv1D
from keras.layers.merge import Concatenate
from keras.layers.convolutional import Conv2D
from keras.layers import concatenate

import sqlite3
def dbsetup(name):
    try:
        connection = sqlite3.connect(name)
    except Exception as e:
        print(e)
        return -1            
    connection.execute('pragma journal_mode=wal')
    return {"cursor" : connection.cursor(), "name" : name, "connection" : connection}


"""
db = dbsetup('R:\\_TrainingServiceBot.db')

batch_size = 1
batchInx = 0
    
y_channels = 1
x_channels = 1  

x = np.zeros((batch_size, 1024, 256, x_channels), dtype=np.float32)        
y = np.zeros((batch_size, 1, 256, y_channels), dtype=np.float32)

record = db["cursor"].execute("SELECT * from Uploads ORDER BY RANDOM() LIMIT 1").fetchone()

blob = record[2]
envbuffer = np.frombuffer(blob, dtype=np.float32)

envbuffer = np.reshape(envbuffer, (925, 300))

#black_mask = np.zeros((921, 300), dtype=np.float32)
#black_mask[np.where((envbuffer > [0.6] ).all(axis = 2))] = 0.9

cv2.imshow('img', envbuffer)
cv2.waitKey(0)

sys.exit()

"""











# build session running on GPU 1
configuration = tf.ConfigProto()
configuration.gpu_options.allow_growth = True
configuration.gpu_options.visible_device_list = "1"
#session = tf.Session(config = configuration)

session = tf.InteractiveSession()
with session.as_default() as session:
    tf.global_variables_initializer().run()

# apply session
keras.backend.set_session(session)




training_batch_size = 2
validation_batch_size = 2
pixel_depth = 8
crop_sizeYdim = 256 #768 #256
crop_sizeXdim = 256 #1024 #256
rescale_labels = True


import keras.layers
import keras.models
import tensorflow as tf

from keras.models import Input, Model
from keras.layers import Conv2D, Concatenate, MaxPooling2D
from keras.layers import UpSampling2D, Dropout, BatchNormalization

from keras.layers import Conv2D
from keras.layers.merge import Concatenate
from keras.layers.convolutional import Conv2D
from keras.layers import concatenate

#session = tf.InteractiveSession()
#with session.as_default() as session:
    #tf.global_variables_initializer().run()

# apply session
#keras.backend.set_session(session)

option_dict_conv = {"activation": "relu", "padding": "same"}
option_dict_bn = {"momentum" : 0.9}


def get_core(dim1, dim2):
    x = keras.layers.Input(shape=(dim1, dim2, 1))
    #xd = Dropout(0.20)(x)
    a = Conv2D(64, kernel_size = (3,3), **option_dict_conv)(x)  
    a = keras.layers.BatchNormalization(**option_dict_bn)(a)
    a = Conv2D(64, kernel_size = (3,3),**option_dict_conv)(a)
    a = keras.layers.BatchNormalization(**option_dict_bn)(a) 
    #a = Dropout(0.2)(a)
    y = keras.layers.pooling.MaxPooling2D()(a)
    b = Conv2D(128, kernel_size = (3,3),**option_dict_conv)(y)
    b = keras.layers.BatchNormalization(**option_dict_bn)(b)
    b = Conv2D(128, kernel_size = (3,3),**option_dict_conv)(b)
    b = keras.layers.BatchNormalization(**option_dict_bn)(b)   
    #b = Dropout(0.2)(b)
    y = keras.layers.pooling.MaxPooling2D()(b)
    c = Conv2D(256, kernel_size = (3,3),**option_dict_conv)(y)
    c = keras.layers.BatchNormalization(**option_dict_bn)(c)
    c = Conv2D(256, kernel_size = (3,3),**option_dict_conv)(c)
    c = keras.layers.BatchNormalization(**option_dict_bn)(c)  
    #c = Dropout(0.2)(c)  
    y = keras.layers.pooling.MaxPooling2D()(c)
    d = Conv2D(512, kernel_size = (3,3),**option_dict_conv)(y)
    d = keras.layers.BatchNormalization(**option_dict_bn)(d)
    d = Conv2D(512, kernel_size = (3,3),**option_dict_conv)(d)
    d = keras.layers.BatchNormalization(**option_dict_bn)(d)  
    #d = Dropout(0.2)(d)  
    d = keras.layers.UpSampling2D()(d)
    y = keras.layers.merge.concatenate([d, c], axis = 3)
    e = Conv2D(256, kernel_size = (3,3),**option_dict_conv)(y)         
    e = keras.layers.BatchNormalization(**option_dict_bn)(e)
    e = Conv2D(256, kernel_size = (3,3),**option_dict_conv)(e)
    e = keras.layers.BatchNormalization(**option_dict_bn)(e)
    #e = Dropout(0.2)(e)
    e = keras.layers.UpSampling2D()(e)   
    y = keras.layers.merge.concatenate([e, b], axis = 3)
    f = Conv2D(128, kernel_size = (3,3),**option_dict_conv)(y)
    f = keras.layers.BatchNormalization(**option_dict_bn)(f)
    f = Conv2D(128, kernel_size = (3,3),**option_dict_conv)(f)
    f = keras.layers.BatchNormalization(**option_dict_bn)(f)
    #f = Dropout(0.2)(f)
    f = keras.layers.UpSampling2D()(f)   
    y = keras.layers.merge.concatenate([f, a], axis = 3)
    y = Conv2D(64, kernel_size = (3,3),**option_dict_conv)(y)
    y = keras.layers.BatchNormalization(**option_dict_bn)(y)
    y = Conv2D(64, kernel_size = (3,3),**option_dict_conv)(y)
    y = keras.layers.BatchNormalization(**option_dict_bn)(y)
    #y = Dropout(0.2)(y)
    return [x, y]
   
   

def get_model(dim1, dim2, activation="relu"):

    [x, y] = get_core(dim1, dim2)
    print(y.shape)
    #y = keras.layers.SeparableConv1D(1, kernel_size = 1024, **{"activation": "relu"} )(y)

    y = keras.layers.Conv2D(1, kernel_size = (1, 1024), **{"activation": "relu"} )(y)

    y = keras.layers.Flatten()(y)

    #y = keras.layers.Dense(256, activation="relu")(y)

    print(y.shape)

    #if activation is not None:
    #    y = keras.layers.Activation(activation)(y)

    model = keras.models.Model(x, y)
    return model




import numpy as np
import skimage.segmentation
import skimage.io
import keras.backend as K
import tensorflow as tf


dim1 = 256
dim2 = 1024

import os
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "models/SpreadToPrice_v48.hdf5"
abs_file_path = os.path.join(script_dir, rel_path)

# 10, 768, 1024, 1
model = get_model(dim1, dim2)
model.load_weights(abs_file_path)  

#model2 = get_model_3_class(dim1, dim2)
#model2.load_weights("LambDefectsUnet_V121.hdf5")   

import glob
#filesList = [fn for fn in glob.glob('C:\\__IntelliscienceData\\__LambDefectProject\\__LambImages\\All_Images_output\\input\\*')]
filesList = None #[fn for fn in glob.glob('C:\\__IntelliscienceData\\__LambDefectProject\\__ImagesOfFriesDeliveredFromLamb\\*')]

import os 
#filesList = [os.path.basename(fn) for fn in filesList]
#filesList = [os.path.basename(fn) for fn in filesList]


import sqlite3
def dbsetup(name):
    try:
        connection = sqlite3.connect(name)
    except Exception as e:
        print(e)
        return -1            
    connection.execute('pragma journal_mode=wal')
    return {"cursor" : connection.cursor(), "name" : name, "connection" : connection}


db = dbsetup('R:\\_TrainingServiceBot.db')

batch_size = 1

y_channels = 1
x_channels = 1  

x = np.zeros((batch_size, 1024, 256, x_channels), dtype=np.float32)        
y = np.zeros((batch_size, 1, 256, y_channels), dtype=np.float32)

for batchInx in range(batch_size):
    
    record = db["cursor"].execute("SELECT * from Uploads ORDER BY RANDOM() LIMIT 1").fetchone()

    blob = record[2]
    envbuffer = np.frombuffer(blob, dtype=np.float32)

    envbuffer = np.reshape(envbuffer, (925, 300))

    #print(envbuffer.shape)

    x[batchInx, 0:921, 0:256, 0] = envbuffer[0:921,0:256]
    y[batchInx, 0:1,   0:256, 0] = envbuffer[922:923,0:256]

    firstClose = y[batchInx, 0,0,0]
    for inx in range(256):
        y[batchInx, 0, inx, 0] = 1000.0 * y[batchInx, 0, inx, 0] / firstClose

x = np.reshape(x, (batch_size, 1024, 256, 1))
x = np.transpose(x, (0,2,1,3))

cv2.imshow('img', x[0])
cv2.waitKey(0)

predictions = model.predict(x, batch_size=1)

y = np.squeeze(y)
predictions = np.squeeze(predictions)

for i in range(len(y)):
    print(y[i]/10, predictions[i]/10)
