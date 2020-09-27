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

import keras.layers
import keras.models
import tensorflow as tf

from keras.models import Input, Model
from keras.layers import Conv2D, Conv1D, Concatenate, MaxPooling2D
from keras.layers import UpSampling2D, Dropout, BatchNormalization

from keras.layers import Conv2D, Conv1D
from keras.layers.merge import Concatenate
from keras.layers.convolutional import Conv2D, Conv1D
from keras.layers import concatenate

#rawImagesDir = r'C:\__IntelliscienceData\AnalyzingData3Images'
#labelImagesDir = r'C:\__IntelliscienceData\AnalyzingData3Labels'

#rawImagesDir = 'C:\__IntelliscienceData\_SingulationLibrary\_analyzing3\AnalyzingData3Images'
#labelImagesDir = 'C:\__IntelliscienceData\_SingulationLibrary\_analyzing3\AnalyzingData3Labels'

#rawImagesDir = r'C:\__IntelliscienceMLCodeAndData\UnetPipeline1TrainingSet\Raw'
#labelImagesDir = r'C:\__IntelliscienceMLCodeAndData\UnetPipeline1TrainingSet\Labels'


import sqlite3
def dbsetup(name):
    try:
        connection = sqlite3.connect(name)
    except Exception as e:
        print(e)
        return -1            
    connection.execute('pragma journal_mode=wal')
    return {"cursor" : connection.cursor(), "name" : name, "connection" : connection}

import os

rawImagesDir = None #= os.path.join(os.getcwd(), 'input')
labelImagesDir = None #= os.path.join(os.getcwd(), 'output')

import glob
trainingFilesList = None #= [fn for fn in glob.glob(rawImagesDir + '\\*.png')]
import os
trainingFilesList = None #= [os.path.basename(fn) for fn in trainingFilesList]


'''
tempList = []
for t in trainingFilesList:
    y_big = cv2.imread( os.path.join(labelImagesDir, t) )
    #if np.isin([0,0,255], y_big).all():
    if np.count_nonzero(   (y_big[:,:] == [0,0,255]).all(axis=2) ) > 1500:
        #cv2.imshow("y_big", y_big)
        #cv2.waitKey(0)
        tempList.append(t)
trainingFilesList = tempList
'''

print("HERRRRRRE\n\n\n")

print(trainingFilesList)

def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

FeatureNotInPicture_Count = 0
def random_sample_generator(x_dir, y_dir, image_names, batch_size, bit_depth, dim1, dim2, rescale_labels):
    global FeatureNotInPicture_Count

    while True:

        db = dbsetup('R:\\_TrainingServiceBot.db')

        #rows = db["cursor"].execute("SELECT sql FROM sqlite_master where type ='table'").fetchall()

        record = db["cursor"].execute("SELECT * from Uploads ORDER BY RANDOM() LIMIT 1").fetchone()

        blob = record[2]
        envbuffer = np.frombuffer(blob, dtype=np.float32)

        requestString = record[1]
        varNames = requestString.split('_')
        #import re
        #m = [m.start() for m in re.finditer(' ', r[0])]
        #varNames = r[0][m[1]:m[2]]
        #tableNames.append(tableName)

        y_channels = 1
        x_channels = 1  
        
        x = np.zeros((batch_size, 1024, 256, x_channels), dtype=np.float32)        
        y = np.zeros((batch_size, 1, 256, y_channels), dtype=np.float32)
            
        for batchInx in range(batch_size):
            

            record = db["cursor"].execute("SELECT * from Uploads ORDER BY RANDOM() LIMIT 1").fetchone()

            blob = record[2]
            envbuffer = np.frombuffer(blob, dtype=np.float32)

            envbuffer = np.reshape(envbuffer, (925, 300))

            #cv2.imshow('img', envbuffer)
            #cv2.waitKey(0)

            #print(envbuffer.shape)

            x[batchInx, 0:921, 0:256, 0] = envbuffer[0:921,0:256]
            y[batchInx, 0:1,   0:256, 0] = envbuffer[922:923,0:256]

            #cv2.imshow('x', x[0,:,:,:])
            #cv2.waitKey(0)            

            firstClose = y[batchInx, 0,0,0]
            for inx in range(256):
                y[batchInx, 0, inx, 0] = 1000.0 * y[batchInx, 0, inx, 0] / firstClose
                #print(y[batchInx, 0, inx, 0])

        db["connection"].close()

        #print(x.shape)

        #ifactor = 1 + np.random.uniform(-0.05, 0.05)
        #x *= ifactor
        x = np.reshape(x, (batch_size, 1024, 256, 1))

        x = np.transpose(x, (0,2,1,3))

        #cv2.imshow('x', x[0,:,:])
        #cv2.waitKey(0) 

        #for i in x[0]:
        #    for j in i:
        #        print(j)
        #import cv2
        #cv2.imshow('Example - Show image in window',x[0])
        #cv2.waitKey(0)

        yield(x, np.reshape(y, (batch_size, 256)))
      


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


print(trainingFilesList)

training_batch_size = 3
validation_batch_size = 1
pixel_depth = 8
crop_sizeYdim = 256 #768 #256
crop_sizeXdim = 256 #1024 #256
rescale_labels = True

train_gen = random_sample_generator(
    rawImagesDir,
    labelImagesDir,
    trainingFilesList,
    training_batch_size,
    pixel_depth,
    crop_sizeYdim,
    crop_sizeXdim,
    rescale_labels
)


import keras.layers
import keras.models
import tensorflow as tf

from keras.models import Input, Model
from keras.layers import Conv2D, Conv1D, Concatenate, MaxPooling2D
from keras.layers import UpSampling2D, Dropout, BatchNormalization

from keras.layers import Conv2D, Conv1D, Lambda
from keras.layers.merge import Concatenate
from keras.layers.convolutional import Conv2D, Conv1D
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

debug = False

def channel_precision(channel, name):
    def precision_func(y_true, y_pred):
        y_pred_tmp = K.cast(tf.equal( K.argmax(y_pred, axis=-1), channel), "float32")
        true_positives = K.sum(K.round(K.clip(y_true[:,:,:,channel] * y_pred_tmp, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred_tmp, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
    
        return precision
    precision_func.__name__ = name
    return precision_func


def channel_recall(channel, name):
    def recall_func(y_true, y_pred):
        y_pred_tmp = K.cast(tf.equal( K.argmax(y_pred, axis=-1), channel), "float32")
        true_positives = K.sum(K.round(K.clip(y_true[:,:,:,channel] * y_pred_tmp, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true[:,:,:,channel], 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
    
        return recall
    recall_func.__name__ = name
    return recall_func

#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
def weighted_crossentropy(y_true, y_pred):
    class_weights = tf.constant([[[[1.0, 1.0]]]])
    #class_weights = tf.constant([[[[15, 1., 2.]]]])
    #class_weights = tf.constant([[[[0.7, 1., 10.]]]])
    #class_weights = tf.constant([[[[.3, 300., 150.]]]])

    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)
    print("y_true")
    print(y_true)
    print("y_pred")
    print(y_pred)
    
    weights = tf.reduce_sum(class_weights * y_true, axis=-1)

    weighted_losses = weights * unweighted_losses

    loss = tf.reduce_mean(weighted_losses)

    return loss


    #dim1 = 384
#dim2 = 512

dim1 = 256 #768 #256
dim2 = 1024 #1024 #256


train_gen = random_sample_generator(
    rawImagesDir,
    labelImagesDir,
    trainingFilesList,
    training_batch_size,
    pixel_depth,
    crop_sizeYdim,
    crop_sizeXdim,
    rescale_labels
)


# 10, 768, 1024, 1
model = get_model(dim1, dim2)

loss = 'mean_squared_error'

metrics = [keras.metrics.categorical_accuracy, 
           channel_recall(channel=0, name="recall"), 
           channel_precision(channel=0, name="precision"),
          ]

#######################################################################################
#######################################################################################
#######################################################################################
optimizer =  keras.optimizers.RMSprop(lr=0.001)
#optimizer = keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, amsgrad=False)

#model.compile(loss=loss, metrics=metrics, optimizer=optimizer)

#model.summary()

# Performance logging
csv_log_file = 'log.csv'
callback_csv = keras.callbacks.CSVLogger(filename=csv_log_file)

callbacks=[callback_csv]



#######################################################################################
#######################################################################################
#######################################################################################
n = 46

import os
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "models/SpreadToPrice_v" + str(n) + ".hdf5"
abs_file_path = os.path.join(script_dir, rel_path)

#model.load_weights(abs_file_path)  

#for layer in model.layers[0:6]:
#    layer.trainable = False

for layer in model.layers:
    print(layer, layer.trainable)


#model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
model.compile(loss=loss, optimizer=optimizer)

model.summary()

for i in range (n,n + 20):
    #model.load_weights("veggieMixModel" + str(i) + ".hdf5")   


    # TRAIN
    statistics = model.fit_generator(
        generator=train_gen,
        steps_per_epoch= 30,
        epochs=2,
        #validation_data=val_gen,
        #validation_steps=1, #int(len(data_partitions["validation"])/config_vars["val_batch_size"]),
        #callbacks=callbacks,
        verbose = 1
    )

    #model_file = "FryModelPipeline2_v5_" + str(i+1) + ".hdf5"
    model_file = "models/SpreadToPrice_v" + str(i+1) + ".hdf5"
    model.save_weights( model_file )


    print('Done! :)') 