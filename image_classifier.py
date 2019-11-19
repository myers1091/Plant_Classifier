# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator 
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras import backend as K

#img_width, img_height = 720, 960
img_width, img_height = 64,64

if K.image_data_format() == 'channels_first':
    input = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
classifier = Sequential()
classifier.add(Conv2D(32,(3,3),input_shape = input_shape,activation = 'relu'))
classifier.add(MaxPooling2D(pool_size= (2,2)))
classifier.add(Flatten())
classifier.add(Dense(units = 128,activation = 'relu'))
classifier.add(Dense(units = 40, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('/Users/john/Documents/DataSci/Leaves/output/train/',
target_size = (64, 64),
batch_size = 32,
class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('/Users/john/Documents/DataSci/Leaves/output/test/',
target_size = (64, 64),
batch_size = 32,
class_mode = 'categorical')

classifier.fit_generator(training_set,steps_per_epoch = 337,
epochs = 25,
validation_data = test_set,
validation_steps = 106)

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('/Users/john/Documents/DataSci/Leaves/P7293143_gr.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
print(result)
