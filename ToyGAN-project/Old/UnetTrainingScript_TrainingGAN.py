###this script attempt to generate inverted images. Discriminator checks whether input image and generator's output image are inversions of each other.


import time
import os
import sys
import random
import numpy as np
import cv2
import glob
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import tensorflow.keras.optimizers
from tensorflow.keras.layers import Input, Conv2D, concatenate, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import UpSampling2D, Dropout, BatchNormalization, Activation
import tensorflow.keras.callbacks
from tensorflow.keras.backend import epsilon
from tensorflow.keras.backend import sum as ksum
from tensorflow.keras.backend import clip as kclip

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import tensorflow.keras.backend as K

import h5py

def str_shape(x):
    return 'x'.join(map(str, x.shape))

#https://gist.github.com/udibr/e522b44a1dc7d3a388d4386d416747f5
def load_weights_custom(model, filepath, lookup={}, ignore=[], transform=None, verbose=True):
    """Modified version of keras load_weights that loads as much as it can.
    Useful for transfer learning.
    read the weights of layers stored in file and copy them to a model layer.
    the name of each layer is used to match the file's layers with the model's.
    It is possible to have layers in the model that dont appear in the file..
    The loading stopps if a problem is encountered and the weights of the
    file layer that first caused the problem are returned.
    # Arguments
        model: Model
            target
        filepath: str
            source hdf5 file
        lookup: dict (optional)
            by default, the weights of each layer in the file are copied to the
            layer with the same name in the model. Using lookup you can replace
            the file name with a different model layer name, or to a list of
            model layer names, in which case the same weights will be copied
            to all layer models.
        ignore: list (optional)
            list of model layer names to ignore in
        transform: None (optional)
            This is an optional function that receives the list of weighs
            read from a layer in the file and the model layer object to which
            these weights should be loaded.
        verbose: bool
            high recommended to keep this true and to follow the print messages.
    # Returns
        weights of the file layer which first caused the load to abort or None
        on successful load.
    """

    flattened_layers = model.layers
    with h5py.File(filepath, mode='r') as f:
        # new file format
        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]

        # we batch weight value assignments in a single backend call
        # which provides a speedup in TensorFlow.
        weight_value_tuples = []
        for name in layer_names:
            print(name)
            #my 1 liner:
            if 'dense' in name:
                continue


            g = f[name]
            weight_names = [n.decode('utf8') for n in
                            g.attrs['weight_names']]
            if len(weight_names):
                weight_values = [g[weight_name] for weight_name in
                                 weight_names]

                target_names = lookup.get(name, name)
                if isinstance(target_names, str):
                    target_names = [target_names]
                # handle the case were lookup asks to send the same weight to multiple layers
                target_names = [target_name for target_name in target_names if
                                target_name == name or target_name not in layer_names]
                for target_name in target_names:

                    try:
                        layer = model.get_layer(name=target_name)
                    except:
                        layer = None
                    if layer:
                        # the same weight_values are copied to each of the target layers
                        symbolic_weights = layer.trainable_weights + layer.non_trainable_weights

                        if transform is not None:
                            transformed_weight_values = transform(weight_values, layer)
                            if transformed_weight_values is not None:
                                weight_values = transformed_weight_values

                        problem = len(symbolic_weights) != len(weight_values)

                        if not problem:
                            weight_value_tuples += zip(symbolic_weights, weight_values)
                    else:
                        problem = True
                    if problem:
                        if not (name in ignore or ignore == '*'):
                            K.batch_set_value(weight_value_tuples)
                            return [np.array(w) for w in weight_values]
                if verbose:
                    print("here")
            else:
                if verbose:
                    print ('skipping this is empty file layer')
        K.batch_set_value(weight_value_tuples)









config = ConfigProto(log_device_placement=True)
#config.gpu_options.allow_growth = True
"""
config.gpu_options.visible_device_list = "0"
session = InteractiveSession(config=config)
with session.as_default() as session:
    tf.global_variables_initializer().run()

# apply session
# keras.backend.set_session(session)
#print(tf.version.VERSION, tf.executing_eagerly(), BatchNormalization._USE_V2_BEHAVIOR)

BatchNormalization._USE_V2_BEHAVIOR = False

tf.compat.v1.disable_eager_execution() # Uncomment this sentence works normally
"""
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

#sys.exit()

def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def randomIllumination(image):
    ifactor = 1 + np.random.uniform(-0.35, 0.0)
    image *= ifactor

    ifactor = 1 + np.random.uniform(-0.42, 0.0)
    image[:,:,0] *= ifactor
        
    ifactor = 1 + np.random.uniform(-0.42, 0.0)
    image[:,:,1] *= ifactor

    ifactor = 1 + np.random.uniform(-0.42, 0.0)
    image[:,:,2] *= ifactor 
    return image


def randomPatch1image(x_big, dim1, dim2):
    start_dim1 = np.random.randint(low=0, high=x_big.shape[0]-dim1)
    start_dim2 = np.random.randint(low=0, high=x_big.shape[1]-dim2)
    x_big  = x_big[start_dim1:start_dim1 + dim1, start_dim2:start_dim2 + dim2] 
    return x_big


def randomPatch(x_big, y_big, dim1, dim2):
    if x_big.shape[0] != dim1:
        start_dim1 = np.random.randint(low=0, high=x_big.shape[0]-dim1)
        start_dim2 = np.random.randint(low=0, high=x_big.shape[1]-dim2)
        x_big  = x_big[start_dim1:start_dim1 + dim1, start_dim2:start_dim2 + dim2] 
        y_big = y_big[start_dim1:start_dim1 + dim1, start_dim2:start_dim2 + dim2] 
    return x_big, y_big    

FeatureNotInPicture_Count = 0
def random_sample_generator___Discriminator_Input_To_Discriminator_Output(trainingFilesList_negatives, trainingFilesList_positives, batch_size, dim1, dim2):
    global FeatureNotInPicture_Count

    while(True):      
        x_channels = 6 
        y_channels = 1
        x = np.zeros((batch_size, dim1, dim2, x_channels), dtype=np.float32)        
        y = np.zeros((batch_size, y_channels), dtype=np.float32)

        x_big_src = np.zeros((dim1, dim2, x_channels), dtype=np.float32)
        x_big_ccl = np.zeros((dim1, dim2, x_channels), dtype=np.float32)

        i = 0
        while(True):

            methodType = 0 #np.random.randint(2)

            if methodType == 0:
                exampleType = np.random.randint(2)               
                
                if exampleType == 1:
                    for ii in range(4):
                        for j in range(4):
                            for k in range(3):
                                x_big_src[ii,j,k] = np.random.randint(255)/255.0                
                    x_big_ccl = 1 - x_big_src                

                elif exampleType == 0:
                    for ii in range(4):
                        for j in range(4):
                            for k in range(3):
                                x_big_src[ii,j,k] = np.random.randint(255)/255.0   
                                x_big_ccl[ii,j,k] = np.random.randint(255)/255.0


                x_big = np.concatenate( (x_big_src, x_big_ccl), axis = 2 )                     

                x[i, :, :, 0:x_channels] = x_big[:,:,0:x_channels]
                y[i,:] = [exampleType]              

            elif methodType == 1 and len(trainingFilesList_negatives) > 0:

                #Train Positive Example Or Train Negative Example
                exampleType = np.random.randint(2)
                
                if exampleType == 0:
                    img_index = np.random.randint(low=0, high = len(trainingFilesList_negatives))                
                    x_big_ccl = cv2.imread(trainingFilesList_positives[trainingFilesList_negatives][:-5] + "o.png")
                    x_big_src = cv2.imread(trainingFilesList_positives[trainingFilesList_negatives][:-5] + "i.png")
                elif exampleType == 1:                
                    # get random image index (to specify which training image in the array of images to select)
                    img_index = np.random.randint(low=0, high = len(trainingFilesList_positives))
                    x_big_ccl = cv2.imread(trainingFilesList_positives[img_index][:-5] + "o.png")
                    x_big_src = cv2.imread(trainingFilesList_positives[img_index][:-5] + "i.png")

                x_big_src = np.float32(x_big_src) / 255.0
                x_big = np.concatenate( (x_big_ccl, x_big_src), axis = 2 )
                            
                x[i, :, :, 0:x_channels] = x_big[:,:,0:x_channels]
                y[i,:] = [exampleType]

            cv2.imshow('x', x[i,:,:,3:])
            cv2.imshow('y', x[i,:,:,0:3])

            cv2.waitKey(1)       
                        
            i = i + 1
            if i >= batch_size:
                break

        # return the buffer
        yield(x, y)

def random_sample_generator___Generator_Input_To_Discriminator_Output(forcedType, trainingFilesList_negatives, trainingFilesList_positives, batch_size, dim1, dim2):
    global FeatureNotInPicture_Count

    while(True):      
        x_channels = 3
        y_channels = 1
        x = np.zeros((batch_size, dim1, dim2, x_channels), dtype=np.float32)        
        y = np.zeros((batch_size, y_channels), dtype=np.float32)

        i = 0
        while(True):

            #Train Positive Example Or Train Negative Example
            exampleType = forcedType

            for ii in range(4):
                for j in range(4):
                    for k in range(3):
                        x[i,ii,j,k] = np.random.randint(255)/255.0

            y[i,:] = [exampleType]

            #cv2.imshow('x', x[i,:,:,0:3])
            #cv2.imshow('y', x[i,:,:,3:])
            #cv2.waitKey(1)       
                        
            i = i + 1
            if i >= batch_size:
                break

        # return the buffer
        yield(x, y)        
        

#option_dict_conv = {"activation": "relu", "padding": "same", "strides" : (1,1)}
option_dict_conv = {"activation": "relu"}
option_dict_bn = {"momentum" : 0.5}

def get_Model_Generator(dim1, dim2):
    dense_layer_size = 1024
    x = Input(shape=( dim1, dim2, 3), name = "original_img")
    a = Flatten()(x)  
    a = Dense(dense_layer_size, **option_dict_conv)(a)
    a = BatchNormalization(momentum = 0.5)(a)
    a = Dense(dense_layer_size, **option_dict_conv)(a)
    a = BatchNormalization(momentum = 0.5)(a)
    a = Dense(dense_layer_size, **option_dict_conv)(a)
    a = BatchNormalization(momentum = 0.5)(a)
    a = Dense(dense_layer_size, **option_dict_conv)(a)
    a = BatchNormalization(momentum = 0.5)(a)
    a = Dense(dense_layer_size, **option_dict_conv)(a)
    a = BatchNormalization(momentum = 0.5)(a)
    a = Dense(dense_layer_size, **option_dict_conv)(a)
    a = BatchNormalization(momentum = 0.5)(a)
    a = Dense(dense_layer_size, **option_dict_conv)(a)
    a = BatchNormalization(momentum = 0.5)(a)
    a = Dense(dense_layer_size, **option_dict_conv)(a)
    a = BatchNormalization(momentum = 0.5)(a)
    a = Dense(dense_layer_size, **option_dict_conv)(a)
    a = BatchNormalization(momentum = 0.5)(a)
    a = Dense(dense_layer_size, **option_dict_conv)(a)    
    a = BatchNormalization(momentum = 0.5)(a)
    a = Dense(dim1*dim2*3, activation = 'sigmoid')(a)
    a = tf.keras.layers.Reshape((dim1, dim2, 3))(a)
    y = concatenate([a, x], axis = 3)  
    model_Generator = tf.keras.Model(inputs = x, outputs = y, name = "generator")   
    return model_Generator

def get_Model_Discriminator(dim1, dim2):
    dense_layer_size = 1024
    x = Input(shape=( dim1, dim2, 6))
    a = Flatten()(x)  
    a = Dense(dense_layer_size, **option_dict_conv)(a)
    a = BatchNormalization(momentum = 0.5)(a)
    a = Dense(dense_layer_size, **option_dict_conv)(a)
    a = BatchNormalization(momentum = 0.5)(a)
    a = Dense(dense_layer_size, **option_dict_conv)(a)
    a = BatchNormalization(momentum = 0.5)(a)
    a = Dense(dense_layer_size, **option_dict_conv)(a)
    a = BatchNormalization(momentum = 0.5)(a)
    a = Dense(dense_layer_size, **option_dict_conv)(a)
    a = BatchNormalization(momentum = 0.5)(a)
    a = Dense(dense_layer_size, **option_dict_conv)(a)
    a = BatchNormalization(momentum = 0.5)(a)
    a = Dense(dense_layer_size, **option_dict_conv)(a)
    a = BatchNormalization(momentum = 0.5)(a)
    a = Dense(dense_layer_size, **option_dict_conv)(a)
    a = BatchNormalization(momentum = 0.5)(a)
    a = Dense(dense_layer_size, **option_dict_conv)(a)
    a = BatchNormalization(momentum = 0.5)(a)
    a = Dense(dense_layer_size, **option_dict_conv)(a)    
    a = BatchNormalization(momentum = 0.5)(a)
    a = Dense(dense_layer_size, **option_dict_conv)(a)    
    a = BatchNormalization(momentum = 0.5)(a)
    y = Dense(1, activation = 'sigmoid')(a)
    model_Discriminator = tf.keras.Model(inputs = x, outputs = y, name = "discriminator")   
    return model_Discriminator

def channel_precision(channel, name):
    def precision_func(y_true, y_pred):
        y_pred_tmp = tf.cast(tf.math.equal( tf.argmax(y_pred, axis=-1), channel), "float32")
        true_positives = ksum(tf.round(kclip(y_true[:,channel] * y_pred_tmp, 0, 1)))
        predicted_positives = ksum(tf.round(kclip(y_pred_tmp, 0, 1)))
        precision = true_positives / (predicted_positives + epsilon())
    
        return precision
    precision_func.__name__ = name
    return precision_func

def channel_recall(channel, name):
    def recall_func(y_true, y_pred):
        y_pred_tmp = tf.cast(tf.math.equal( tf.argmax(y_pred, axis=-1), channel), "float32")
        true_positives = ksum(tf.round(kclip(y_true[:,channel] * y_pred_tmp, 0, 1)))
        possible_positives = ksum(tf.round(kclip(y_true[:,channel], 0, 1)))
        recall = true_positives / (possible_positives + epsilon())
    
        return recall
    recall_func.__name__ = name
    return recall_func

def weighted_crossentropy(y_true, y_pred):
    class_weights = tf.constant([[[[1.0, 1.0]]]])
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)    
    weights = tf.reduce_sum(class_weights * y_true, axis=-1)
    weighted_losses = weights * unweighted_losses
    loss = tf.reduce_mean(weighted_losses)
    return loss

def reset_weights(model):    
    session = K.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'): 
            layer.kernel.initializer.run(session=session)
        if hasattr(layer, 'bias_initializer'):
            layer.bias.initializer.run(session=session)    

dim1 = int(4)
dim2 = int(4)
#[model_Generator, model_Discriminator] = get_model()

model_Generator = get_Model_Generator(dim1 = dim1, dim2 = dim2)
model_Discriminator = get_Model_Discriminator(dim1 = dim1, dim2 = dim2)
#for layer in model_Discriminator.layers:
#    layer.trainable = False



generator_input = Input(shape=( dim1, dim2, 3), name = "img")
discriminator_input = model_Generator(generator_input) # discriminator input is same as generator output
discriminator_output = model_Discriminator(discriminator_input)
model_GAN = tf.keras.Model(inputs = generator_input, outputs = discriminator_output, name = "GAN")

model_Generator.summary()
model_Discriminator.summary()
model_GAN.summary()

#To start training from a previous model weights, specify model number with n.

n = 1
"""
modelFileRootName = 'CCLModel_GAN_v'
relative_file_path = os.path.join('models', modelFileRootName + str(n) + '.hdf5')
if os.path.exists(relative_file_path):
    print("Loading Model From Model File ", relative_file_path)
    model_GAN.load_weights(relative_file_path)  
"""
#turn off certain layers from being trained (another training trick that can sometimes help).
#for layer in model_Discriminator.layers:
#    layer.trainable = False

#model_GAN.layers[2].trainable = False
for layer in model_GAN.layers:
    print(layer, layer.trainable)

#model_GAN.layers[2].load_weights("models/CCLModel_Discriminator2_v2.hdf5")
#model_GAN.layers[2].trainable = False



"""
weights = []
import h5py
with h5py.File("Temp_Discriminator_Weights.hdf5", 'r') as hdf5_f:
    namesList = hdf5_f.attrs['layer_names']
    print(namesList)
    for n in namesList:        
        layername = n.decode("utf-8")
        if 'dense' in layername:
            continue
        print(layername)
        group = hdf5_f[ layername ]        
        for p_name in group.keys():
            param = group[p_name]
            weights.append(param)
"""


def custom_loss(y_true, y_pred):

    y_pred_clipped = kclip(y_pred, 1e-8, 1-1e-8)
    log_lik = -K.log(1 - abs(y_true - y_pred_clipped))
    loss = tf.reduce_mean(log_lik)
    return loss

#Train
if True:
    #for i in range (n,n + 1):
    while True:

        #reset_weights(model_Generator)
        #model_GAN.layers[1].set_weights(initial_weights)

        ##################################################################################################################
        ##################################################################################################################
        ## Train Discriminator
        ##################################################################################################################

        #if os.path.exists("Temp_Discriminator_Weights.hdf5"):
        #    model_GAN.layers[2].load_weights("Temp_Discriminator_Weights.hdf5")   

        model_GAN.layers[1].trainable = False
        model_GAN.layers[2].trainable = True
        model_GAN.layers[2].compile(
            #loss = custom_loss, 
            #loss = 'mean_squared_error',
            loss = tf.keras.losses.BinaryCrossentropy(),
            #optimizer = tf.keras.optimizers.RMSprop(lr=0.001) #specifiez the learning rate
            #metrics = [tf.keras.metrics.categorical_accuracy, 
            #    channel_recall(channel=0, name="Class_0_recall"), 
            #    channel_precision(channel=0, name="Class_0_precision"),
            #    channel_recall(channel=1, name="Class_1_recall"), 

            #    channel_precision(channel=1, name="Class_1_precision")],     
            optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
        )
     
        for i in range (10):
            statistics = model_GAN.layers[2].fit(
                x = random_sample_generator___Discriminator_Input_To_Discriminator_Output(
                                trainingFilesList_negatives = [fn for fn in glob.glob('TrainingImages_Negatives\\*.png')],
                                trainingFilesList_positives = [fn for fn in glob.glob('TrainingImages_Positives\\*.png')],
                                batch_size = 50,
                                dim1 = dim1,
                                dim2 = dim2),
                steps_per_epoch = 200,
                epochs = 1,
                verbose = 1
            )
            #Save new weights to file
            model_GAN.layers[2].save_weights("Temp_Discriminator_Weights.hdf5") 

        ##################################################################################################################
        ##################################################################################################################
        ## Train Generator
        ##################################################################################################################

        #trick to reinitialize generator's weights.
        model_GAN.layers[1].set_weights ( get_Model_Generator(dim1 = dim1, dim2 = dim2).get_weights())
        model_GAN.layers[1].trainable = True
        model_GAN.layers[2].trainable = False
        model_GAN.compile(
            #loss = custom_loss, 
            #loss = 'mean_squared_error',
            loss = tf.keras.losses.BinaryCrossentropy(),
            #optimizer = tf.keras.optimizers.RMSprop(lr=0.001) #specifiez the learning rate
            #metrics = [tf.keras.metrics.categorical_accuracy, 
            #    channel_recall(channel=0, name="Class_0_recall"), 
            #    channel_precision(channel=0, name="Class_0_precision"),
            #    channel_recall(channel=1, name="Class_1_recall"), 
            #    channel_precision(channel=1, name="Class_1_precision")],     
            optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
        )               
        for i in range(1):
            statistics = model_GAN.fit(
                x = random_sample_generator___Generator_Input_To_Discriminator_Output(
                                forcedType = 1,
                                trainingFilesList_negatives = [fn for fn in glob.glob('TrainingImages_Negatives\\*.png')],
                                trainingFilesList_positives = [fn for fn in glob.glob('TrainingImages_Positives\\*.png')],
                                batch_size = 50,
                                dim1 = dim1,
                                dim2 = dim2),
                steps_per_epoch = 300,
                epochs = 1,
                verbose = 1#,
                #callbacks=callbacks
            )
            #model_GAN.layers[1].save_weights("Temp_Generator_Weights.hdf5")  
            print(i)
            print(i)
            print(i)
            print(i)
            print(i)
        
        """
        ##################################################################################################################
        ##################################################################################################################
        ## Train Discriminator
        ##################################################################################################################

        model_GAN.layers[1].trainable = False
        model_GAN.layers[2].trainable = True
        model_GAN.compile(
            #loss = weighted_crossentropy, 
            #loss = tf.keras.losses.CategoricalCrossentropy(),
            loss = tf.keras.losses.BinaryCrossentropy(),
            #optimizer = tf.keras.optimizers.RMSprop(lr=0.001) #specifiez the learning rate
            #metrics = [tf.keras.metrics.categorical_accuracy, 
            #    channel_recall(channel=0, name="Class_0_recall"), 
            #    channel_precision(channel=0, name="Class_0_precision"),
            #    channel_recall(channel=1, name="Class_1_recall"), 
            #    channel_precision(channel=1, name="Class_1_precision")],     
            optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
        )        
        
        statistics = model_GAN.fit(
            x = random_sample_generator___Generator_Input_To_Discriminator_Output(
                            forcedType = 0,
                            trainingFilesList_negatives = [fn for fn in glob.glob('TrainingImages_Negatives\\*.png')],
                            trainingFilesList_positives = [fn for fn in glob.glob('TrainingImages_Positives\\*.png')],
                            batch_size = 200,
                            # neural network Trains on 256x256 image tile
                            dim1 = dim1,
                            dim2 = dim2),
            steps_per_epoch = 8,
            epochs = 1,
            verbose = 1
        )        
        model_GAN.layers[2].save_weights("Temp_Discriminator_Weights.hdf5")  
        """

        ##################################################################################################################
        ##################################################################################################################
        ## Store Generator Images
        ##################################################################################################################

        print("saving generator images")
        for i in range(10):
            [x,y_true] = random_sample_generator___Generator_Input_To_Discriminator_Output(
                            forcedType = 1,
                            trainingFilesList_negatives = [fn for fn in glob.glob('TrainingImages_Negatives\\*.png')],
                            trainingFilesList_positives = [fn for fn in glob.glob('TrainingImages_Positives\\*.png')],
                            batch_size = 1,
                            dim1 = dim1,
                            dim2 = dim2).__next__()

            generator_pred = model_GAN.layers[1].predict(x).squeeze()

            inputImg = generator_pred[:,:,3:]
            generatedImg = generator_pred[:,:,:3]

            f = "TrainingImages_Negatives\\" + str(int(time.time())) + "-" + str(i)
            cv2.imshow("intput", inputImg)
            cv2.imshow("output", generatedImg)
            cv2.waitKey(1)
            cv2.imwrite(f + "_i.png", inputImg*255.0)
            cv2.imwrite(f + "_o.png", generatedImg*255.0)

            y_GAN_out = model_GAN.predict(x).squeeze()
        
        #print (y_GAN_out)
        #cv2.imshow('y_pred', y_pred[:,:,:3])
        #cv2.waitKey(1)  
        # 
        #break 

print('Done! :)') 