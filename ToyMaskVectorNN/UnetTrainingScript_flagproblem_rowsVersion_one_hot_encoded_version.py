import os
import random

import numpy as np

import keras.backend
import keras.callbacks
import keras.layers
import keras.models
import keras.optimizers

from keras.models import Input, Model
from keras.layers import UpSampling2D, Dropout, BatchNormalization
from keras.layers import Conv2D, Conv1D, Lambda, Concatenate, MaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers import concatenate
import tensorflow as tf

print("Tensorflow Version: ", tf.__version__)

# build session running on GPU 1
configuration = tf.ConfigProto()
configuration.gpu_options.allow_growth = True
configuration.gpu_options.visible_device_list = "1"

session = tf.InteractiveSession()
with session.as_default() as session:
    tf.global_variables_initializer().run()

# apply session
keras.backend.set_session(session)

def random_sample_generator(batch_size, rowCount):
    while True:
        x_in = np.zeros((batch_size, rowCount, 2), dtype=np.float32)        
        y_out = np.zeros((batch_size, rowCount), dtype=np.float32)
        
        numbersColumn = 0
        flagsColumn = 1

        for batchInx in range(batch_size):

            randOffset = np.random.randint(low=0, high=10)

            for c in range(rowCount):
                x_in[batchInx, c, numbersColumn] =  c#np.random.randint(low=0, high=10)

            flagged_row = np.random.randint(low=0, high=rowCount)
            #print(flagged_row)
            #flagging one of the rows
            x_in[batchInx, flagged_row, flagsColumn] = 1.0
            #storing flagged row's column 0 number as the output
            y_out[batchInx, flagged_row] = 1.0# x_in[batchInx, flagged_row, numbersColumn]

        yield(x_in, y_out)

def get_model(dim1, dim2):
    option_dict_conv = {"activation": "relu", "padding": "same"}
    option_dict_bn = {"momentum" : 0.9}
    x = keras.layers.Input(shape=(dim1, dim2))
    y = Conv1D(256, kernel_size = (1), **option_dict_conv)(x)      
    y = keras.layers.Flatten()(y)  
    y = keras.layers.BatchNormalization(**option_dict_bn)(y)    
    y = keras.layers.Dense(256, activation="relu")(y)
    y = keras.layers.BatchNormalization(**option_dict_bn)(y)
    y = keras.layers.Dense(256, activation="relu")(y)
    y = keras.layers.BatchNormalization(**option_dict_bn)(y) 
    y = keras.layers.Dense(dim1, activation="softmax")(y)

    model = keras.models.Model(input=[x], output=[y])
    return model

rowCount = 256
model = get_model(dim1 = rowCount, dim2 = 2)
model.summary()

n = 0
abs_file_path = os.path.join( os.path.dirname(__file__) , "model_" + str(n) + ".hdf5")
if os.path.exists(abs_file_path):
    print("Loading Model From Model File ", abs_file_path)
    model.load_weights(abs_file_path)  

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.RMSprop(lr=0.00001))
model.summary()

#for layer in model.layers[1:]:
#    layer.trainable = False

for i in range (n,n + 15):
    statistics = model.fit_generator(
        generator = random_sample_generator(batch_size = 400, rowCount = rowCount),
        steps_per_epoch = 200,
        epochs = 1,
        verbose = 1
    )
    #model_file = "model_" + str(i+1) + ".hdf5"
    #model.save_weights( model_file )


if False:
    for layer in model.layers: 
        if True: #type(layer) is keras.layers.Conv1D:
            #m = np.array(layer.get_weights())

            weights = layer.get_weights()

            if weights:
                print("all weights")
                print(weights)


            #for index in itertools.combinations_with_replacement(range(9, -1, -1), 3):
            #kernelshape is x,y, input channel depth, kernelCount aka output depth

            #https://docs.python.org/3.7/tutorial/floatingpoint.html
            """
            for kernel in range(0,layerKernelShape[3]):            
                for y in range(0,layerKernelShape[0]):
                    for x in range(0,layerKernelShape[1]):   
                        for channel in range(0,layerKernelShape[2]): 
                            floatnumber = weights[y][x][channel][kernel]
                            print("weight: ", floatnumber)
                            
            for kernel in range(0,layerKernelShape[3]):
                floatnumber = biases[kernel] 
                print("bias: ", floatnumber)                        
            """
            #print(layer.name)
            #print(type(layer.get_weights()[0][0][0][0][0]))
            #print(layer.get_weights()[0][0][0][0][0], layer.get_weights()[0][0][1][0][0], layer.get_weights()[0][0][2][0][0])
            #print(layer.get_weights()[0][1][0][0][0], layer.get_weights()[0][1][1][0][0], layer.get_weights()[0][1][2][0][0])
            #print(layer.get_weights()[0][2][0][0][0], layer.get_weights()[0][2][1][0][0], layer.get_weights()[0][2][2][0][0])

[x,y_actual] = random_sample_generator(batch_size = 1, rowCount = rowCount).__next__()
y_pred = model.predict(x)

print("\ninference test")
print("input")
print(x)
print("y_pred")
print(np.argmax(y_pred))
print("y_actual")
print(np.argmax(y_actual))

