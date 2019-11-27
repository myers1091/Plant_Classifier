import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator 

from keras.models import load_model
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('/home/johncs/Documents/Plant_Classifier/colorleaves/train/',
target_size = (64, 64),
batch_size = 32,
class_mode = 'categorical')

model = load_model('/home/johncs/Documents/Plant_Classifier/weights/weights-03-0.0553_2.h5')
test_image = image.load_img('colorleaves/test/tilia_americana/1252646647_0003.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
training_set.class_indices
print(result)