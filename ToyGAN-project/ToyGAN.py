# GAN Program Description:
# The GAN's Input is a number between 0 and 1.  
# The GAN's generator outputs both the input number and 1 minus the input number.  
# The GAN's discriminator determines whether the two numbers sum to 1.

#What's interesting is that in many GAN solutions, the generator's input is noise that doesn't bear a recognizable relationship to the generator's output. In this particular example, the generator's input and output are related.

#Keras's functional api is utilized in this program. 
#Keras's sequential API, more common among novices, is not used.
#https://keras.io/guides/functional_api/

import numpy as np
import tensorflow.keras.optimizers
from tensorflow.keras.layers import BatchNormalization, Input, concatenate, Flatten, Dense
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def random_sample_generator___Generator_Input_To_Discriminator_Output(forced_example_type, batch_size):
    while(True):      
        x = np.random.random(size=(batch_size, 1))
        y = np.full( shape=(batch_size, 1), fill_value = forced_example_type)
        yield(x, y) 

def random_sample_generator___Discriminator_Input_To_Discriminator_Output(batch_size):
    while(True):      
        x = np.zeros((batch_size, 2), dtype=np.float32)        
        y = np.zeros(batch_size, dtype=np.float32)

        for i in range(batch_size):
            exampleType = np.random.randint(2)               
            
            if exampleType == 1:
                x_src = np.random.random()
                x_out = 1 - x_src                            
            else:
                x_src = np.random.random()
                x_out = np.random.random()

            x[i] = [x_src, x_out]
            y[i] = exampleType

        yield(x, y)       
        
def get_Model_Generator():

    dense_layer_size = 8
    momentumVal = 0.99
    init_xavier = tf.glorot_uniform_initializer()

    x = Input(shape=(1))
    hidden_layer = Dense(dense_layer_size, activation = 'relu', kernel_initializer = init_xavier)(x)
    #hidden_layer = BatchNormalization(momentum = momentumVal)(hidden_layer)
    hidden_layer = Dense(dense_layer_size, activation = 'relu', kernel_initializer = init_xavier)(hidden_layer)
    #hidden_layer = BatchNormalization(momentum = momentumVal)(hidden_layer)
    hidden_layer = Dense(dense_layer_size, activation = 'relu', kernel_initializer = init_xavier)(hidden_layer)
    #hidden_layer = BatchNormalization(momentum = momentumVal)(hidden_layer)
    hidden_layer = Dense(dense_layer_size, activation = 'relu', kernel_initializer = init_xavier)(hidden_layer)
    #hidden_layer = BatchNormalization(momentum = momentumVal)(hidden_layer)
    y = Dense(1, activation = 'sigmoid')(hidden_layer)

    model_Generator = tf.keras.Model(inputs = x, outputs = concatenate([x, y]), name = "generator")   
    return model_Generator

def get_Model_Discriminator():

    dense_layer_size = 8
    momentumVal = 0.99
    init_xavier = tf.glorot_uniform_initializer()

    x = Input(shape=(2))
    hidden_layer = Dense(dense_layer_size, activation = 'relu', kernel_initializer = init_xavier)(x)
    #hidden_layer = BatchNormalization(momentum = momentumVal)(hidden_layer)
    hidden_layer = Dense(dense_layer_size, activation = 'relu', kernel_initializer = init_xavier)(hidden_layer)
    #hidden_layer = BatchNormalization(momentum = momentumVal)(hidden_layer)
    hidden_layer = Dense(dense_layer_size, activation = 'relu', kernel_initializer = init_xavier)(hidden_layer)
    #hidden_layer = BatchNormalization(momentum = momentumVal)(hidden_layer)
    hidden_layer = Dense(dense_layer_size, activation = 'relu', kernel_initializer = init_xavier)(hidden_layer)
    #hidden_layer = BatchNormalization(momentum = momentumVal)(hidden_layer)
    y = Dense(1, activation = 'sigmoid')(hidden_layer)
    
    model_Discriminator = tf.keras.Model(inputs = x, outputs = y, name = "discriminator")   
    return model_Discriminator

# creating two neural net models and linking them together by 
# feeding the output of one into the input of the other, to form model_GAN below
model_Generator = get_Model_Generator()
model_Discriminator = get_Model_Discriminator()

generator_input = Input(shape=( 1))
generator_output = model_Generator(generator_input) 
discriminator_input = generator_output
discriminator_output = model_Discriminator(discriminator_input)
model_GAN = tf.keras.Model(inputs = generator_input, outputs = discriminator_output, name = "GAN")

model_Generator.summary()
model_Discriminator.summary()
model_GAN.summary()

##################################################################################################################
## Train Discriminator

model_GAN.layers[2].compile(
    #loss = 'mean_squared_error', 
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0005)
)

statistics = model_GAN.layers[2].fit(
    x = random_sample_generator___Discriminator_Input_To_Discriminator_Output(batch_size = 1400),
    steps_per_epoch = 3500,
    epochs = 1,
    verbose = 1
)

print("Comparing discriminator inference with ground truth")
for i in range(30):
    [x,y_true] = random_sample_generator___Discriminator_Input_To_Discriminator_Output(
                    batch_size = 1).__next__()
    y_pred = model_GAN.layers[2].predict(x).squeeze()
    x = x.squeeze()
    print("Discriminator inputs: ", x, "\tSum:", "{0:.2f}".format(x[0] + x[1]), "\ty_true: ", y_true, "\ty_pred: ", y_pred)

##################################################################################################################
# Train Generator

model_GAN.layers[1].trainable = True
model_GAN.layers[2].trainable = False
model_GAN.compile(
    #loss = 'mean_squared_error', 
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0005)
)               

statistics = model_GAN.fit(
    x = random_sample_generator___Generator_Input_To_Discriminator_Output(forced_example_type = 1,batch_size = 100000),
    steps_per_epoch = 250,
    epochs = 100,
    verbose = 1
)

##################################################################################################################
print("Printing Generator Input/Output Examples")
for i in range(30):
    [x,y_true] = random_sample_generator___Generator_Input_To_Discriminator_Output(
                    forced_example_type = 1,
                    batch_size = 1).__next__()
    generator_pred = model_GAN.layers[1].predict(x).squeeze()
    print("Input: ", x, "\tOutputs: ","{0:.7f}".format(generator_pred[1]), "{0:.7f}".format(generator_pred[0]), 
        "\tOutputs Summed: ", "{0:.7f}".format(generator_pred[1]+generator_pred[0]))

print('Done! :)') 