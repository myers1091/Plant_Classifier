 # Importing the Keras libraries and packages
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator 
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
#from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow 
from keras import backend as K
from keras.callbacks import EarlyStopping, TensorBoard,ModelCheckpoint
import datetime
import argparse
from ResNet import ResNet50
"""
Code Blocks taken from Priyanka Dwivedi:
https://github.com/priya-dwivedi/Deep-Learning/blob/master/resnet_keras/Residual_Networks_yourself.ipynb
"""
# def identity_block(X, f, filters, stage, block):
#     """

#     Implementation of the identity block
    
#     Arguments:
#     X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
#     f -- integer, specifying the shape of the middle CONV's window for the main path
#     filters -- python list of integers, defining the number of filters in the CONV layers of the main path
#     stage -- integer, used to name the layers, depending on their position in the network
#     block -- string/character, used to name the layers, depending on their position in the network
    
#     Returns:
#     X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
#     """
    
#     # defining name basis
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'
    
#     # Retrieve Filters
#     F1, F2, F3 = filters
    
#     # Save the input value. You'll need this later to add back to the main path. 
#     X_shortcut = X
    
#     # First component of main path
#     X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
#     X = Activation('relu')(X)

    
#     # Second component of main path (≈3 lines)
#     X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
#     X = Activation('relu')(X)

#     # Third component of main path (≈2 lines)
#     X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

#     # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
#     X = Add()([X, X_shortcut])
#     X = Activation('relu')(X)
    
    
#     return X
# def convolutional_block(X, f, filters, stage, block, s = 2):

#     """
#     Implementation of the convolutional block as defined in Figure 4
    
#     Arguments:
#     X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
#     f -- integer, specifying the shape of the middle CONV's window for the main path
#     filters -- python list of integers, defining the number of filters in the CONV layers of the main path
#     stage -- integer, used to name the layers, depending on their position in the network
#     block -- string/character, used to name the layers, depending on their position in the network
#     s -- Integer, specifying the stride to be used
    
#     Returns:
#     X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
#     """
    
#     # defining name basis
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'
    
#     # Retrieve Filters
#     F1, F2, F3 = filters
    
#     # Save the input value
#     X_shortcut = X


#     ##### MAIN PATH #####
#     # First component of main path 
#     X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
#     X = Activation('relu')(X)

#     # Second component of main path (≈3 lines)
#     X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
#     X = Activation('relu')(X)


#     # Third component of main path (≈2 lines)
#     X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)


#     ##### SHORTCUT PATH #### (≈2 lines)
#     X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1',
#                         kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
#     X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

#     # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
#     X = Add()([X, X_shortcut])
#     X = Activation('relu')(X)
    
    
#     return X

# def ResNet50(input_shape=(64, 64, 3), classes=6):
#     """
#     Implementation of the popular ResNet50 the following architecture:
#     CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
#     -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

#     Arguments:
#     input_shape -- shape of the images of the dataset
#     classes -- integer, number of classes

#     Returns:
#     model -- a Model() instance in Keras
#     """

#     # Define the input as a tensor with shape input_shape
#     X_input = Input(input_shape)

#     # Zero-Padding
#     X = ZeroPadding2D((3, 3))(X_input)

#     # Stage 1
#     X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
#     X = BatchNormalization(axis=3, name='bn_conv1')(X)
#     X = Activation('relu')(X)
#     X = MaxPooling2D((3, 3), strides=(2, 2))(X)

#     # Stage 2
#     X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
#     X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
#     X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

#     ### START CODE HERE ###

#     # Stage 3 (≈4 lines)
#     X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
#     X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
#     X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
#     X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

#     # Stage 4 (≈6 lines)
#     X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
#     X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
#     X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
#     X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
#     X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
#     X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

#     # Stage 5 (≈3 lines)
#     X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
#     X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
#     X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

#     # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
#     X = AveragePooling2D((2,2), name="avg_pool")(X)

#     ### END CODE HERE ###

#     # output layer
#     X = Flatten()(X)
#     X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
#     # Create model
#     model = Model(inputs = X_input, outputs = X, name='ResNet50')

#     return model
##Argparser to take weight from trained model
parser = argparse.ArgumentParser(description='Basic Arg parser')
parser.add_argument("--weight", default=1, type=str, help="This is the weight file")
args = parser.parse_args()
trainedweight = args.weight



#Rescale images to 64x64 (needs to be square)
img_width, img_height = 64,64

#get the ordering if the image formatting right
if K.image_data_format() == 'channels_first':
    input = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#Create resnet50 model
#
#check if --weight option was given
if type(trainedweight) == str: #if str, then pull weight from saved file
    model = load_model('weights/weights-05-0.0621_1.h5')
else: #if not option called, trainedweights == int and train from scratch
    model = ResNet50(input_shape = input_shape, classes = 184)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('./colorleaves/train/',
target_size = (64, 64),
batch_size = 32,
class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('./colorleaves/val/',
target_size = (64, 64),
batch_size = 32,
class_mode = 'categorical')

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=0,write_graph=True, write_images=True)
early_stopping_monitor = EarlyStopping(monitor='val_loss', mode='min',patience=2)
#define the checkpoint
filepath = '/home/johncs/Documents/Plant_Classifier/weights/weights-{epoch:02d}-{loss:.4f}_2.h5'
checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')

model.fit_generator(training_set,steps_per_epoch = 6109,
epochs = 25,
validation_data = test_set,
validation_steps = 693,
callbacks=[tensorboard_callback,early_stopping_monitor,checkpoint])



# import numpy as np
# from keras.preprocessing import image
# test_image = image.load_img('/Users/john/Documents/DataSci/Leaves/P7293143_gr.jpg', target_size = (64, 64))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis = 0)
# result = classifier.predict(test_image)
# training_set.class_indices
# print(result)
