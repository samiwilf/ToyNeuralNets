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
        x = np.reshape(x, (batch_size, 1024, 256))

        x = np.transpose(x, (0,2,1))

        #cv2.imshow('x', x[0,:,:])
        #cv2.waitKey(0) 

        #for i in x[0]:
        #    for j in i:
        #        print(j)
        #import cv2
        #cv2.imshow('Example - Show image in window',x[0])
        #cv2.waitKey(0)

        yield(x, np.reshape(y, (batch_size, 256)))
        
        '''
        #temp = np.zeros((batch_size, 768, 1024, 2), dtype=np.float32)
        # get one image at a time
        i = 0
        while(True):
                       
            # get random image
            img_index = np.random.randint(low=0, high=n_images)
            
            # open images
            #print(image_names[img_index])
            x_big = cv2.imread(os.path.join(x_dir, image_names[img_index])) 
            x_big = np.float32(x_big)
            x_big = x_big * rescale_factor
            x_big = np.float32(x_big)
            #print (y_dir)
            #print(image_names[img_index])
            #print(type(x_big))
            #print(x_big.dtype)
            

            #y_big = cv2.imread(os.path.join(y_dir, image_names[img_index][:-4] + '_pre.png'))
            #print(image_names[img_index][:-4] + '.png')
            y_big = cv2.imread(os.path.join(y_dir, image_names[img_index][:-4] + '.png'))

            black_mask = np.zeros((y_big.shape[0], y_big.shape[1]), dtype=np.float32)    
            count = np.count_nonzero(   (y_big[:,:] == [231,191,200]).all(axis=2) )
            if count > 200:
                black_mask[np.where((y_big == [231,191,200] ).all(axis = 2))] = [1]
            else:
                black_mask[np.where((y_big != [0, 0, 0] ).all(axis = 2))] = [1]

            temp = np.zeros((y_big.shape[0], y_big.shape[1], 2), dtype=np.float32)
            temp[:,:,1] = black_mask
            temp[:,:,0] = abs(1.0-black_mask[:,:])
            y_big = temp       
            
            #print(type(y_big))
            
            #resizing
            #random_angle = np.random.uniform(0, 360)
            #x_big = rotateImage(x_big, random_angle)
            #y_big = rotateImage(y_big, random_angle)
            #y_big[np.where((y_big <= [0.98,0.98]).all(axis=2))] = [1,0]

            # get random crop
            start_dim1 = np.random.randint(low=0, high=x_big.shape[0]-dim1)
            start_dim2 = np.random.randint(low=0, high=x_big.shape[1]-dim2)
            
            #start_dim1 = x_big.shape[0]-dim1
            #start_dim2 = x_big.shape[1]-dim2

            patch_x  = x_big[start_dim1:start_dim1 + dim1, start_dim2:start_dim2 + dim2] 
            patch_y = y_big[start_dim1:start_dim1 + dim1, start_dim2:start_dim2 + dim2] 
            #print("\nhere\n")

            #used for rotation
            #if np.count_nonzero(   (patch_x[:,:] <= [0,0,0]).all(axis=2) ) > 200:
            #    print(np.count_nonzero(   (patch_x[:,:] <= [0,0,0]).all(axis=2) ))
            #    continue

            if False:
                whitePixelCount = np.count_nonzero(   (patch_y[:,:] == [0,1]).all(axis=2) )
                if whitePixelCount < 7000:
                    FeatureNotInPicture_Count += 1
                else:
                    FeatureNotInPicture_Count = 0
                if FeatureNotInPicture_Count > 1:
                    #print(np.count_nonzero(   (patch_x[:,:] <= [0,0,0]).all(axis=2) ))
                    continue    

            #print(patch_y.shape)
            if(do_augmentation):
                
                rand_flip = np.random.randint(low=0, high=2)
                rand_rotate = np.random.randint(low=0, high=4)
                
                # flip
                if(rand_flip == 0):
                    patch_x = np.flip(patch_x, 0)
                    patch_y = np.flip(patch_y, 0)
                
                # rotate
                for rotate_index in range(rand_rotate):
                    patch_x = np.rot90(patch_x)
                    patch_y = np.rot90(patch_y)


                # illumination
                ifactor = 1 + np.random.uniform(-0.03, 0.0)
                patch_x *= ifactor

                ifactor = 1 + np.random.uniform(-0.02, 0.0)
                patch_x[:,:,0] *= ifactor
                    
                ifactor = 1 + np.random.uniform(-0.02, 0.0)
                patch_x[:,:,1] *= ifactor

                ifactor = 1 + np.random.uniform(-0.02, 0.0)
                patch_x[:,:,2] *= ifactor 
                    
            x[i, :, :, 0:x_channels] = patch_x[:,:,0:x_channels]
            y[i, :, :, 0:y_channels] = patch_y[:,:,0:y_channels]           
                    
            cv2.imshow('x', x[i])
            #cv2.waitKey(1)
            cv2.imshow('y', y[i,:,:, 0])
            #cv2.waitKey(1)
            #cv2.imshow('y_big2', patch_y[:,:,1])
            cv2.waitKey(1)       

            #x = patch_x[:,:,0:x_channels]
            #y = patch_y[:,:,0:y_channels]
            #x = x.reshape(i, x.shape[0], x.shape[1], x.shape[2])
            #y = y.reshape(i, y.shape[0], y.shape[1], y.shape[2])
            i = i + 1
            if i >= batch_size:
                break

        # return the buffer
        yield(x, y)
        '''




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

training_batch_size = 5
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

mm = 2
def get_core(dim1, dim2):
    x = keras.layers.Input(shape=(dim1, dim2))
    #xd = Dropout(0.20)(x)
    a = Conv1D(64*mm, kernel_size = (3), **option_dict_conv)(x)  
    a = keras.layers.BatchNormalization(**option_dict_bn)(a)
    a = Conv1D(64*mm, kernel_size = (3),**option_dict_conv)(a)
    a = keras.layers.BatchNormalization(**option_dict_bn)(a) 
    #a = Dropout(0.2)(a)
    y = keras.layers.pooling.MaxPooling1D()(a)
    b = Conv1D(128*mm, kernel_size = (3),**option_dict_conv)(y)
    b = keras.layers.BatchNormalization(**option_dict_bn)(b)
    b = Conv1D(128*mm, kernel_size = (3),**option_dict_conv)(b)
    b = keras.layers.BatchNormalization(**option_dict_bn)(b)   
    #b = Dropout(0.2)(b)
    y = keras.layers.pooling.MaxPooling1D()(b)
    c = Conv1D(256*mm, kernel_size = (3),**option_dict_conv)(y)
    c = keras.layers.BatchNormalization(**option_dict_bn)(c)
    c = Conv1D(256*mm, kernel_size = (3),**option_dict_conv)(c)
    c = keras.layers.BatchNormalization(**option_dict_bn)(c)  
    #c = Dropout(0.2)(c)  
    y = keras.layers.pooling.MaxPooling1D()(c)
    d = Conv1D(512*mm, kernel_size = (3),**option_dict_conv)(y)
    d = keras.layers.BatchNormalization(**option_dict_bn)(d)
    d = Conv1D(512*mm, kernel_size = (3),**option_dict_conv)(d)
    d = keras.layers.BatchNormalization(**option_dict_bn)(d)  
    #d = Dropout(0.2)(d)  
    d = keras.layers.UpSampling1D()(d)
    y = keras.layers.merge.concatenate([d, c], axis = 2)
    e = Conv1D(256*mm, kernel_size = (3),**option_dict_conv)(y)         
    e = keras.layers.BatchNormalization(**option_dict_bn)(e)
    e = Conv1D(256*mm, kernel_size = (3),**option_dict_conv)(e)
    e = keras.layers.BatchNormalization(**option_dict_bn)(e)
    #e = Dropout(0.2)(e)
    e = keras.layers.UpSampling1D()(e)   
    y = keras.layers.merge.concatenate([e, b], axis = 2)
    f = Conv1D(128*mm, kernel_size = (3),**option_dict_conv)(y)
    f = keras.layers.BatchNormalization(**option_dict_bn)(f)
    f = Conv1D(128*mm, kernel_size = (3),**option_dict_conv)(f)
    f = keras.layers.BatchNormalization(**option_dict_bn)(f)
    #f = Dropout(0.2)(f)
    f = keras.layers.UpSampling1D()(f)   
    y = keras.layers.merge.concatenate([f, a], axis = 2)
    y = Conv1D(64*mm, kernel_size = (3),**option_dict_conv)(y)
    y = keras.layers.BatchNormalization(**option_dict_bn)(y)
    y = Conv1D(64*mm, kernel_size = (3),**option_dict_conv)(y)
    y = keras.layers.BatchNormalization(**option_dict_bn)(y)
    #y = Dropout(0.2)(y)
    return [x, y]
   

def get_model(dim1, dim2, activation="relu"):

    [x, y] = get_core(dim1, dim2)
    print(y.shape)
    #y = keras.layers.SeparableConv1D(1, kernel_size = 1024, **{"activation": "relu"} )(y)

    y = keras.layers.Conv1D(1, kernel_size = (1), **{"activation": "relu"} )(y)

    y = keras.layers.Flatten()(y)

    #y = keras.layers.Dense(256, activation="relu")(y)

    print(y.shape)

    #if activation is not None:
    #    y = keras.layers.Activation(activation)(y)

    model = keras.models.Model(x, y)
    return model

'''
def get_core(dim1, dim2):
    x = keras.layers.Input(shape=(dim1, dim2))
    #x = keras.layers.BatchNormalization(**option_dict_bn)(x)

    # xSliced = Lambda(lambda x : x[:,0:8,0:8])(x)

    # return [x, xSliced]


    a = Conv1D(64, kernel_size = (3), **option_dict_conv)(x)  
    a = keras.layers.BatchNormalization(**option_dict_bn)(a)
    a = Conv1D(64, kernel_size = (3), **option_dict_conv)(a)
    a = keras.layers.BatchNormalization(**option_dict_bn)(a) 
    y = keras.layers.pooling.MaxPooling1D()(a)
    b = Conv1D(128, kernel_size = (3), **option_dict_conv)(y)
    b = keras.layers.BatchNormalization(**option_dict_bn)(b)
    b = Conv1D(128, kernel_size = (3), **option_dict_conv)(b)
    b = keras.layers.BatchNormalization(**option_dict_bn)(b)   
    y = keras.layers.pooling.MaxPooling1D()(b)
    c = Conv1D(256, kernel_size = (3), **option_dict_conv)(y)
    c = keras.layers.BatchNormalization(**option_dict_bn)(c)
    c = Conv1D(256, kernel_size = (3), **option_dict_conv)(c)
    c = keras.layers.BatchNormalization(**option_dict_bn)(c)    
    y = keras.layers.pooling.MaxPooling1D()(c)
    d = Conv1D(512, kernel_size = (3), **option_dict_conv)(y)
    d = keras.layers.BatchNormalization(**option_dict_bn)(d)
    d = Conv1D(512, kernel_size = (3), **option_dict_conv)(d)
    d = keras.layers.BatchNormalization(**option_dict_bn)(d)    
    #d = keras.layers.UpSampling2D()(d)

    #########################################################

    y = keras.layers.pooling.MaxPooling1D()(d)
    e = Conv1D(512, kernel_size = (3), **option_dict_conv)(y)
    e = keras.layers.BatchNormalization(**option_dict_bn)(e)
    e = Conv1D(512, kernel_size = (3), **option_dict_conv)(e)
    e = keras.layers.BatchNormalization(**option_dict_bn)(e)    
    #d = keras.layers.UpSampling2D()(d)


    y = keras.layers.pooling.MaxPooling1D()(e)
    f = Conv1D(512, kernel_size = (3), **option_dict_conv)(y)
    f = keras.layers.BatchNormalization(**option_dict_bn)(f)
    f = Conv1D(512, kernel_size = (3), **option_dict_conv)(f)
    f = keras.layers.BatchNormalization(**option_dict_bn)(f)    
    #d = keras.layers.UpSampling2D()(d)

    y = keras.layers.pooling.MaxPooling1D()(f)
    g = Conv1D(512, kernel_size = (3), **option_dict_conv)(y)
    g = keras.layers.BatchNormalization(**option_dict_bn)(g)
    g = Conv1D(512, kernel_size = (3), **option_dict_conv)(g)
    g = keras.layers.BatchNormalization(**option_dict_bn)(g)    
    #d = keras.layers.UpSampling2D()(d)



    #y = keras.layers.pooling.MaxPooling2D()(g)
    #h = Conv2D(2048, kernel_size = (3), **option_dict_conv)(y)
    ##d = keras.layers.BatchNormalization(**option_dict_bn)(d)
    #h = Conv2D(2048, kernel_size = (3), **option_dict_conv)(h)
    ##d = keras.layers.BatchNormalization(**option_dict_bn)(d)    
    ##d = keras.layers.UpSampling2D()(d)



    ##slice dd to merge with ddd to form ee
    #gSliced = Lambda(lambda g : g[:,int(0/16):int(dim1/128),int(0/16):int(dim2/128),:])(c)
    #y = keras.layers.merge.concatenate([h, gSliced], axis = 3)
    #h = Conv2D(256, kernel_size = (3), **option_dict_conv)(y)         
    #e = keras.layers.BatchNormalization(**option_dict_bn)(e)
    #h = Conv2D(256, kernel_size = (3), **option_dict_conv)(h)





    ##slice dd to merge with ddd to form ee
    fSliced = Lambda(lambda f : f[:, 0:int(dim1/64), :])(f)
    y = keras.layers.merge.concatenate([g, fSliced], axis = 2)
    i = Conv1D(256, kernel_size = (3), **option_dict_conv)(y)         
    i = keras.layers.BatchNormalization(**option_dict_bn)(i)
    i = Conv1D(256, kernel_size = (3), **option_dict_conv)(i)
    i = keras.layers.BatchNormalization(**option_dict_bn)(i)

    ##slice d to merge with ee to form eee
    eSliced = Lambda(lambda e : e[:, 0:int(dim1/64), :])(e)
    y = keras.layers.merge.concatenate([i, eSliced], axis = 2)
    j = Conv1D(256, kernel_size = (3), **option_dict_conv)(y)         
    j = keras.layers.BatchNormalization(**option_dict_bn)(j)
    j = Conv1D(256, kernel_size = (3), **option_dict_conv)(j)
    j = keras.layers.BatchNormalization(**option_dict_bn)(j)


    ##slice d to merge with ee to form eee
    dSliced = Lambda(lambda d : d[:, 0:int(dim1/64), :])(d)
    y = keras.layers.merge.concatenate([j, dSliced], axis = 2)
    k = Conv1D(256, kernel_size = (3), **option_dict_conv)(y)         
    k = keras.layers.BatchNormalization(**option_dict_bn)(k)
    k = Conv1D(256, kernel_size = (3), **option_dict_conv)(k)    
    k = keras.layers.BatchNormalization(**option_dict_bn)(k)

    #################################################


    cSliced = Lambda(lambda c : c[:, 0:int(dim1/64), :])(c)
    y = keras.layers.merge.concatenate([k, cSliced], axis = 2)
    l = Conv1D(256, kernel_size = (3), **option_dict_conv)(y)         
    l = keras.layers.BatchNormalization(**option_dict_bn)(l)
    l = Conv1D(256, kernel_size = (3), **option_dict_conv)(l)
    l = keras.layers.BatchNormalization(**option_dict_bn)(l)    
    #e = keras.layers.BatchNormalization(**option_dict_bn)(e)
    #e = keras.layers.UpSampling2D()(e)   

    bSliced = Lambda(lambda b : b[:, 0:int(dim1/64), :])(b)

    y = keras.layers.merge.concatenate([l, bSliced], axis = 2)
    m = Conv1D(128, kernel_size = (3), **option_dict_conv)(y)
    m = keras.layers.BatchNormalization(**option_dict_bn)(m)
    m = Conv1D(128, kernel_size = (3), **option_dict_conv)(m)
    m = keras.layers.BatchNormalization(**option_dict_bn)(m)
    #f = keras.layers.UpSampling2D()(f)   

    aSliced = Lambda(lambda a : a[:, 0:int(dim1/64), :])(a)

    y = keras.layers.merge.concatenate([m, aSliced], axis = 2)
    n = Conv1D(128, kernel_size = (3), **option_dict_conv)(y)
    n = keras.layers.BatchNormalization(**option_dict_bn)(n)
    n = Conv1D(128, kernel_size = (3), **option_dict_conv)(n)
    n = keras.layers.BatchNormalization(**option_dict_bn)(n)    
    #f = keras.layers.BatchNormalization(**option_dict_bn)(f)
    #f = keras.layers.UpSampling2D()(f)      

    y = n
    #y = keras.layers.BatchNormalization(**option_dict_bn)(y)
    return [x, y]
   
   

def get_model_3_class(dim1, dim2, activation="softmax"):

    [x, y] = get_core( dim1 , dim2)

    #y = keras.layers.Convolution2D(3, 1, 1, **option_dict_conv)(y)

    #if activation is not None:
    #    y = keras.layers.Activation(activation)(y)
    y = keras.layers.Flatten()(y)
    y = keras.layers.Dense(3, activation="softmax")(y)

    model = keras.models.Model(x, y)
    return model

'''


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

    '''
    print("WRITING DAT FILE!!!")
    print("WRITING DAT FILE!!!")
    print("WRITING DAT FILE!!!")
    print("WRITING DAT FILE!!!")
    print("WRITING DAT FILE!!!")
    print("WRITING DAT FILE!!!")
    print("WRITING DAT FILE!!!")
    print("WRITING DAT FILE!!!")
    print("WRITING DAT FILE!!!")
    print("WRITING DAT FILE!!!")
    print("WRITING DAT FILE!!!")
    print("WRITING DAT FILE!!!")

    import struct

    f = open(r"models/GreenBeansModel_v" + str(i+1) + ".dat", 'wb')

    import itertools

    for layer in model.layers: 
        print("layer output shape", layer.output_shape)
        print( int(layer.output_shape[1]),int(layer.output_shape[2]), layer.output_shape[3] )

    for layer in model.layers: 
        if type(layer) is keras.layers.Conv2D:
            #m = np.array(layer.get_weights())

            weights = layer.get_weights()[0]
            biases = layer.get_weights()[1]

            layerKernelShape = layer.get_weights()[0].shape
            print("layerKernelShape", layerKernelShape)
            f.write( struct.pack('i', layerKernelShape[0] ))
            f.write( struct.pack('i', layerKernelShape[1] ))
            f.write( struct.pack('i', layerKernelShape[2] ))
            f.write( struct.pack('i', layerKernelShape[3] ))

            print("layer intput shape", layer.input_shape)
            print("layer output shape", layer.output_shape)

            #for index in itertools.combinations_with_replacement(range(9, -1, -1), 3):
            #kernelshape is x,y, input channel depth, kernelCount aka output depth

            #https://docs.python.org/3.7/tutorial/floatingpoint.html
            for kernel in range(0,layerKernelShape[3]):            
                for y in range(0,layerKernelShape[0]):
                    for x in range(0,layerKernelShape[1]):   
                        for channel in range(0,layerKernelShape[2]): 
                            floatnumber = weights[y][x][channel][kernel]
                            f.write( struct.pack('f', floatnumber) )
            for kernel in range(0,layerKernelShape[3]):
                floatnumber = biases[kernel]
                f.write( struct.pack('f', floatnumber) )                            

            #print(layer.name)
            #print(type(layer.get_weights()[0][0][0][0][0]))
            #print(layer.get_weights()[0][0][0][0][0], layer.get_weights()[0][0][1][0][0], layer.get_weights()[0][0][2][0][0])
            #print(layer.get_weights()[0][1][0][0][0], layer.get_weights()[0][1][1][0][0], layer.get_weights()[0][1][2][0][0])
            #print(layer.get_weights()[0][2][0][0][0], layer.get_weights()[0][2][1][0][0], layer.get_weights()[0][2][2][0][0])


    for layer in model.layers: 
        mylayersweights = layer.get_weights()
        #print(len(mylayersweights))
        if len(mylayersweights) == 4:

            gamma =  K.eval(layer.gamma)
            beta = K.eval(layer.beta)
            moving_mean = K.eval(layer.moving_mean)
            moving_variance = K.eval(layer.moving_variance)
            epsilon = layer.epsilon

            #print(mylayersweights)
            f.write( struct.pack('i', len(gamma) ))
            for e in gamma:
                f.write( struct.pack('f', e) )
            f.write( struct.pack('i', len(beta) ))
            for e in beta:
                f.write( struct.pack('f', e) )
            f.write( struct.pack('i', len(moving_mean) ))
            for e in moving_mean:
                f.write( struct.pack('f', e) )
            f.write( struct.pack('i', len(moving_variance) ))
            for e in moving_variance:
                f.write( struct.pack('f', e) )      
            f.write( struct.pack('i', len(moving_variance) ))
            for e in moving_variance:
                f.write( struct.pack('f', e) )                                                                


            #print('gamma:          ', len(gamma) )
            #print('beta:           ', len(beta) )
            #print('moving_mean:    ', len(moving_mean) )
            #print('moving_variance:', len(moving_variance) )
            #print('epsilon:        ', len(epsilon) )
            #print('data_in:        ', data_in)
            #print('result:         ', result)

    epsilon = model.layers[2].epsilon
    f.write( struct.pack('f', epsilon) )

    f.close()     
    '''

    print('Done! :)') 