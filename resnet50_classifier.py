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
from keras.utils.np_utils import to_categorical 
from ResNet import ResNet50, load_dataset

##Argparser to take weight from trained model
parser = argparse.ArgumentParser(description='Basic Arg parser')
parser.add_argument("--weight", default=1, type=str, help="This is the weight file")
args = parser.parse_args()
trainedweight = args.weight

#load hdf5 datafile
#returns X_train, y_train, X_val, y_val, data shape (#images, image width, image height, 3)
hdf5_path = './data/dataset.hdf5'
X_train, y_train_origin, X_val, y_val_origin, data_shape, nb_class= load_dataset(hdf5_path,subtract_mean = 0)

# Convert training and test labels to one hot matrices
y_train = to_categorical(y_train_origin, num_classes=nb_class)
y_val   = to_categorical(y_val_origin, num_classes= nb_class)

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_val.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(y_train.shape))
print ("X_test shape: " + str(X_val.shape))
print ("Y_test shape: " + str(y_val.shape))

#grab image 
img_width = data_shape[1]; img_height = data_shape[2]
#get the ordering if the image formatting right
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#Create resnet50 model
#
#check if --weight option was given
if type(trainedweight) == str: #if str, then pull weight from saved file
    model = load_model(trainedweight)
else: #if not option called, trainedweights == int and train from scratch
    model = ResNet50(input_shape = input_shape, classes = nb_class)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



# train_datagen = ImageDataGenerator(rescale = 1./255,
# shear_range = 0.2,
# zoom_range = 0.2,
# horizontal_flip = True)
# test_datagen = ImageDataGenerator(rescale = 1./255)
# training_set = train_datagen.flow_from_directory('./colorleaves/train/',
# target_size = (64, 64),
# batch_size = 32,
# class_mode = 'categorical')
# test_set = test_datagen.flow_from_directory('./colorleaves/val/',
# target_size = (64, 64),
# batch_size = 32,
# class_mode = 'categorical')

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1,write_graph=True, write_images=True)
early_stopping_monitor = EarlyStopping(monitor='val_loss', mode='min',patience=2)
#define the checkpoint
filepath = './weights/weights-'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'{epoch:02d}-{loss:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')
#
# 
# need to change this from hard coded to dynamic
# model.fit_generator(training_set,steps_per_epoch = 6109,
# epochs = 25,
# validation_data = test_set,
# validation_steps = 693,
# callbacks=[tensorboard_callback,early_stopping_monitor,checkpoint])

#New setup to run using images stored in hdf5 and processed outside of main code
model.fit(X_train, y_train, epochs = 25, batch_size = 32,validation_data=(X_val,y_val),callbacks=[tensorboard_callback,early_stopping_monitor,checkpoint],shuffle="batch")



# import numpy as np
# from keras.preprocessing import image
# test_image = image.load_img('/Users/john/Documents/DataSci/Leaves/P7293143_gr.jpg', target_size = (64, 64))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis = 0)
# result = classifier.predict(test_image)
# training_set.class_indices
# print(result)
