from random import shuffle
import glob
import numpy as np
import h5py
import cv2


#############
#Grab images and create a list of labels
#############
#Define train, val, and test path
headdir = './test/'
train = headdir +'train/'
val = headdir+ 'val/'
test = headdir + 'test/'
#Give path for output h5 file
hdf5_path = headdir + 'dataset.hdf5'
#create lsit for labels
train_labels = []
val_labels = []
test_labels = []
#lists of images (needs to be sorted)
train_images = sorted(glob.glob(train+'**/*'))
val_images = sorted(glob.glob(val+'**/*'))
test_images = sorted(glob.glob(test+'**/*'))

#dummy index for classes
i = 0; j = 0; k = 0

#loop through sorted train and val directories and create list of labels
for planttype in sorted(glob.glob(train+'*')):
    for image in glob.glob(planttype+'/*'):
        train_labels.append(i)
    i = i + 1
for planttype in sorted(glob.glob(val+'*')):
    for image in glob.glob(planttype+'/*'):
        val_labels.append(j)
    j = j + 1
for planttype in sorted(glob.glob(test+'*')):
    for image in glob.glob(planttype+'/*'):
        test_labels.append(k)
    k = k + 1

###############
#Now that we have the images and labels, we need to save them to a h5
###############

data_order = 'tf'  # 'th' for Theano, 'tf' for Tensorflow
# check the order of data and chose proper data shape to save images
if data_order == 'th':
    train_shape = (len(train_images), 3, 224, 224)
    val_shape = (len(val_images), 3, 224, 224)
    test_shape = (len(test_images), 3, 224, 224)
elif data_order == 'tf':
    train_shape = (len(train_images), 224, 224, 3)
    val_shape = (len(val_images), 224, 224, 3)
    test_shape = (len(test_images), 224, 224, 3)

# open a hdf5 file and create earrays
hdf5_file = h5py.File(hdf5_path, mode='w')
hdf5_file.create_dataset("train_img", train_shape, np.int16)
hdf5_file.create_dataset("val_img", val_shape, np.int16)
hdf5_file.create_dataset("test_img", test_shape, np.int16)
hdf5_file.create_dataset("train_mean", train_shape[1:], np.float32)
hdf5_file.create_dataset("train_labels", (len(train_images),), np.int16)
hdf5_file["train_labels"][...] = train_labels
hdf5_file.create_dataset("val_labels", (len(val_images),), np.int16)
hdf5_file["val_labels"][...] = val_labels
hdf5_file.create_dataset("test_labels", (len(test_images),), np.int16)
hdf5_file["test_labels"][...] = test_labels

# a numpy array to save the mean of the images
mean = np.zeros(train_shape[1:], np.float32)
# loop over train addresses
for i in range(len(train_images)):
    # print how many images are saved every 1000 images
    if i % 1000 == 0 and i > 1:
        print('Train data: {}/{}'.format(i, len(train_images)))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = train_images[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # add any image pre-processing here
    # if the data order is Theano, axis orders should change
    if data_order == 'th':
        img = np.rollaxis(img, 2)
    # save the image and calculate the mean so far
    hdf5_file["train_img"][i, ...] = img[None]
    mean += img / float(len(train_labels))
# loop over validation addresses
for i in range(len(val_images)):
    # print how many images are saved every 1000 images
    if i % 1000 == 0 and i > 1:
        print('Validation data: {}/{}'.format(i, len(val_images)))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = val_images[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # add any image pre-processing here
    # if the data order is Theano, axis orders should change
    if data_order == 'th':
        img = np.rollaxis(img, 2)
    # save the image
    hdf5_file["val_img"][i, ...] = img[None]
# loop over test addresses
for i in range(len(test_images)):
    # print how many images are saved every 1000 images
    if i % 1000 == 0 and i > 1:
        print('Test data: {}/{}'.format(i, len(test_images)))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = test_images[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # add any image pre-processing here
    # if the data order is Theano, axis orders should change
    if data_order == 'th':
        img = np.rollaxis(img, 2)
    # save the image
    hdf5_file["test_img"][i, ...] = img[None]
# save the mean and close the hdf5 file
hdf5_file["train_mean"][...] = mean
hdf5_file.close()



