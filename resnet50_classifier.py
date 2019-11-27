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
    model = load_model(trainedweight)
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
filepath = './weights/weights-{epoch:02d}-{loss:.4f}_2.h5'
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
