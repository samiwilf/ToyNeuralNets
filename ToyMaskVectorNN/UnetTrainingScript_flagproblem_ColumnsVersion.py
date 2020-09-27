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


# build session running on GPU 1
configuration = tf.ConfigProto()
configuration.gpu_options.allow_growth = True
configuration.gpu_options.visible_device_list = "1"

session = tf.InteractiveSession()
with session.as_default() as session:
    tf.global_variables_initializer().run()

# apply session
keras.backend.set_session(session)


def random_sample_generator(batch_size, columnCount):
    while True:
        x_in = np.zeros((batch_size, 2, columnCount, 1), dtype=np.float32)        
        y_out = np.zeros((batch_size), dtype=np.float32)
        
        numbersRow = 0
        flagsRow = 1

        for batchInx in range(batch_size):

            for c in range(columnCount):
                x_in[batchInx, numbersRow, c, 0] =  np.random.randint(low=0, high=99999)

            flagged_column = np.random.randint(low=0, high=columnCount)

            #flagging one of the rows
            x_in[batchInx, flagsRow, flagged_column, 0] = 1.0
            #storing flagged row's column 0 number as the output
            y_out[batchInx] = x_in[batchInx, numbersRow, flagged_column, 0]

        yield(x_in, y_out)


def get_model(dim1, dim2):
    option_dict_conv = {"activation": "relu"}

    x = keras.layers.Input(shape=(dim1, dim2, 1))
    y = Conv2D(1, kernel_size = (2, 1), **option_dict_conv)(x)  
    y = keras.layers.Flatten()(y)  
    y = keras.layers.Dense(1, activation="relu")(y)

    model = keras.models.Model(x, y)
    return model

columnCount = 8
dim1 = 2
dim2 = columnCount
model = get_model(dim1, dim2)
model.summary()

csv_log_file = 'log.csv'
callback_csv = keras.callbacks.CSVLogger(filename=csv_log_file)
callbacks=[callback_csv]

n = 0

import os
script_dir = os.path.dirname(__file__) 
rel_path = "model_" + str(n) + ".hdf5"
abs_file_path = os.path.join(script_dir, rel_path)
#model.load_weights(abs_file_path)  

loss = 'mean_squared_error'
optimizer =  keras.optimizers.RMSprop(lr=0.01)
model.compile(loss=loss, optimizer=optimizer)

training_batch_size = 400
train_gen = random_sample_generator(training_batch_size, columnCount)

for i in range (n,n + 20):

    statistics = model.fit_generator(
        generator=train_gen,
        steps_per_epoch= 500,
        epochs=2,
        verbose = 1
    )

    model_file = "model_" + str(i+1) + ".hdf5"
    model.save_weights( model_file )