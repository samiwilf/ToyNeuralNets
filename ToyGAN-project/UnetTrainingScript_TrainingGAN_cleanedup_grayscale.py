import time
import os
import sys
import random
import numpy as np
import cv2
import glob
import tensorflow.keras.optimizers
from tensorflow.keras.layers import Input, Conv2D, concatenate, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import UpSampling2D, Dropout, BatchNormalization, Activation
import tensorflow.keras.callbacks
from tensorflow.keras.backend import epsilon
from tensorflow.keras.backend import sum as ksum
from tensorflow.keras.backend import clip as kclip
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow.keras.backend as K

tf.disable_v2_behavior()
config = ConfigProto(log_device_placement=True)
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

def random_sample_generator___Discriminator_Input_To_Discriminator_Output(batch_size, dim1, dim2):
    while(True):      
        x_channels = 2 
        y_channels = 1
        x = np.zeros((batch_size, dim1, dim2, x_channels), dtype=np.float32)        
        y = np.zeros((batch_size, y_channels), dtype=np.float32)

        for i in range(batch_size):

            exampleType = np.random.randint(2)               
            
            if exampleType == 1:
                x_src = np.random.random(size=(dim1, dim2, 1))
                x_out = 1 - x_src                
            
            elif exampleType == 0:
                x_src = np.random.random(size=(dim1, dim2, 1))
                x_out = np.random.random(size=(dim1, dim2, 1))

            #print(x_src[0,0], x_out[0,0])
            x[i] = np.concatenate( (x_src, x_out), axis = 2 ) 
            y[i] = [exampleType]

        yield(x, y)

def random_sample_generator___Generator_Input_To_Discriminator_Output(forcedType, batch_size, dim1, dim2):
    while(True):      

        x = np.random.random(size=(batch_size, dim1, dim2, 1))
        y = np.full( shape=(batch_size, 1), fill_value = forcedType)

        yield(x, y)        
        
def get_Model_Generator(dim1, dim2):
    dense_layer_size = 1024
    x = Input(shape=( dim1, dim2, 1), name = "original_img")
    a = Flatten()(x)  
    a = Dense(dense_layer_size, activation = 'relu')(a)
    a = BatchNormalization(momentum = 0.5)(a)
    a = Dense(dense_layer_size, activation = 'relu')(a)
    a = BatchNormalization(momentum = 0.5)(a)
    a = Dense(dense_layer_size, activation = 'relu')(a)
    a = BatchNormalization(momentum = 0.5)(a)
    a = Dense(dense_layer_size, activation = 'relu')(a)
    a = BatchNormalization(momentum = 0.5)(a)
    a = Dense(dense_layer_size, activation = 'relu')(a)
    a = BatchNormalization(momentum = 0.5)(a)
    a = Dense(dense_layer_size, activation = 'relu')(a)
    a = BatchNormalization(momentum = 0.5)(a)
    a = Dense(dense_layer_size, activation = 'relu')(a)
    a = BatchNormalization(momentum = 0.5)(a)
    a = Dense(dense_layer_size, activation = 'relu')(a)
    a = BatchNormalization(momentum = 0.5)(a)    
    a = Dense( dim1 * dim2 * 1, activation = 'sigmoid')(a)
    a = tf.keras.layers.Reshape((dim1, dim2, 1))(a)
    y = concatenate([a, x], axis = 3)  
    model_Generator = tf.keras.Model(inputs = x, outputs = y, name = "generator")   
    return model_Generator

def get_Model_Discriminator(dim1, dim2):
    dense_layer_size = 1024
    x = Input(shape=( dim1, dim2, 2))
    a = Flatten()(x)  
    a = Dense(dense_layer_size, activation = 'relu')(a)
    a = BatchNormalization(momentum = 0.5)(a)
    a = Dense(dense_layer_size, activation = 'relu')(a)
    a = BatchNormalization(momentum = 0.5)(a)
    a = Dense(dense_layer_size, activation = 'relu')(a)
    a = BatchNormalization(momentum = 0.5)(a)
    a = Dense(dense_layer_size, activation = 'relu')(a)
    a = BatchNormalization(momentum = 0.5)(a)
    a = Dense(dense_layer_size, activation = 'relu')(a)
    a = BatchNormalization(momentum = 0.5)(a)
    a = Dense(dense_layer_size, activation = 'relu')(a)
    a = BatchNormalization(momentum = 0.5)(a)
    a = Dense(dense_layer_size, activation = 'relu')(a)
    a = BatchNormalization(momentum = 0.5)(a)
    a = Dense(dense_layer_size, activation = 'relu')(a)
    a = BatchNormalization(momentum = 0.5)(a)          
    y = Dense(1, activation = 'sigmoid')(a)
    model_Discriminator = tf.keras.Model(inputs = x, outputs = y, name = "discriminator")   
    return model_Discriminator

dim1 = int(1)
dim2 = int(1)

generator_input = Input(shape=( dim1, dim2, 1), name = "img")
model_Generator = get_Model_Generator(dim1 = dim1, dim2 = dim2)
model_Discriminator = get_Model_Discriminator(dim1 = dim1, dim2 = dim2)

discriminator_input = model_Generator(generator_input) 
discriminator_output = model_Discriminator(discriminator_input)
model_GAN = tf.keras.Model(inputs = generator_input, outputs = discriminator_output, name = "GAN")

model_Generator.summary()
model_Discriminator.summary()
model_GAN.summary()

##################################################################################################################
## Train Discriminator
model_GAN.layers[2].load_weights("Temp_Discriminator_Weights.hdf5") #If you want to reuse last run

if False:
    model_GAN.layers[1].trainable = False
    model_GAN.layers[2].trainable = True
    model_GAN.layers[2].compile(
        loss = 'mean_squared_error', #tf.keras.losses.BinaryCrossentropy(), 
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
    )
    statistics = model_GAN.layers[2].fit(
        x = random_sample_generator___Discriminator_Input_To_Discriminator_Output(
                        batch_size = 1000,
                        dim1 = dim1,
                        dim2 = dim2),
        steps_per_epoch = 2500,
        epochs = 1,
        verbose = 1
    )
    model_GAN.layers[2].save_weights("Temp_Discriminator_Weights.hdf5") 

for i in range(100):
    [x,y_true] = random_sample_generator___Discriminator_Input_To_Discriminator_Output(
                    batch_size = 1,
                    dim1 = dim1,
                    dim2 = dim2).__next__()
    generator_pred = model_GAN.layers[2].predict(x).squeeze()
    x = x.squeeze()
    print(x, x[0] + x[1], y_true)
    print(generator_pred)
    print("\n")

##################################################################################################################
# Train Generator

model_GAN.layers[1].trainable = True
model_GAN.layers[2].trainable = False
model_GAN.compile(
    loss = 'mean_squared_error', #tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.5)
)               

statistics = model_GAN.fit(
    x = random_sample_generator___Generator_Input_To_Discriminator_Output(
                    forcedType = 1,
                    batch_size = 10000,
                    dim1 = dim1,
                    dim2 = dim2),
    steps_per_epoch = 250,
    epochs = 1,
    verbose = 1
)

##################################################################################################################
print("Printing Generator Input/Output Examples")

if not os.path.exists('GAN_Images'):
    os.makedirs('GAN_Images')
for i in range(100):
    [x,y_true] = random_sample_generator___Generator_Input_To_Discriminator_Output(
                    forcedType = 1,
                    batch_size = 1,
                    dim1 = dim1,
                    dim2 = dim2).__next__()
    generator_pred = model_GAN.layers[1].predict(x).squeeze()
    print(generator_pred[1], generator_pred[0] )

print('Done! :)') 