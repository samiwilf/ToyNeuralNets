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


def random_sample_generator(batch_size, rowCount):
    while True:
        x_in = np.zeros((batch_size, rowCount, 2), dtype=np.float32)        
        y_out = np.zeros((batch_size), dtype=np.float32)
        
        numbersColumn = 0
        flagsColumn = 1

        for batchInx in range(batch_size):

            for c in range(rowCount):
                x_in[batchInx, c, numbersColumn] =  np.random.randint(low=0, high=99999)

            flagged_row = np.random.randint(low=0, high=rowCount)

            #flagging one of the rows
            x_in[batchInx, flagged_row, flagsColumn] = 1.0
            #storing flagged row's column 0 number as the output
            y_out[batchInx] = x_in[batchInx, flagged_row, numbersColumn]

        yield(x_in, y_out)


def get_model(dim1, dim2):
    option_dict_conv = {"activation": "relu", "padding": "same"}

    x = keras.layers.Input(shape=(dim1, dim2))
    y = Conv1D(1, kernel_size = (1), **option_dict_conv)(x)  
    y = keras.layers.Flatten()(y)  
    y = keras.layers.Dense(1, activation="relu")(y)

    model = keras.models.Model(x, y)
    return model

rowCount = 8
dim1 = rowCount
dim2 = 2
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
model.summary()

for i in range (n,n + 20):

    statistics = model.fit_generator(
        generator=random_sample_generator(batch_size = 400, rowCount = rowCount),
        steps_per_epoch= 500,
        epochs=2,
        verbose = 1
    )

    model_file = "model_" + str(i+1) + ".hdf5"
    model.save_weights( model_file )