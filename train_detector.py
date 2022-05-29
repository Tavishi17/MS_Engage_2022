# importing the required packages
from tensorflow.keras.layers import GlobalMaxPool2D,AveragePooling2D,Dropout,Flatten,Dense,Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from imutils import paths
import numpy as np
import argparse
import os

# constructing the argument parser and passign arguments
ap=argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True)
args = vars(ap.parse_args())
# learning rate
LR = 0.0004
# number of eopchs to train for and batch size
epochs = 25
Batch_Size = 128
image_size = (224,224)

#-----Loading images through the dataset 
# and doing required image processing and 
# to make it compatible with the input expected by MobileNet-----

# Accessing Data
imagePaths = list(paths.list_images(args["dataset"]))
dataset = []
labels = []

# loop over the image paths
for imagePath in imagePaths:
	# extracting the class label from the filename
	label = imagePath.split(os.path.sep)[-2]
	# loading the input image and preprocessing it
	image = load_img(imagePath, target_size=image_size)
	image = img_to_array(image)
	image = preprocess_input(image)
	# updating the data and labels lists
	dataset.append(image)
	labels.append(label)

# converting the data and labels to NumPy arrays
dataset = np.array(dataset)
labels = np.array(labels)

# initialising label binarizer and one-hot encoding the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)


# splitting the data for training and validation into 80-20 ratio
(train_X,valid_X,train_Y,valid_Y) = train_test_split(dataset, labels,
	test_size = 0.20, stratify = labels, random_state = 42)

# constructing the training image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range = 20,
	zoom_range = 0.15,
	width_shift_range = 0.2,
	height_shift_range = 0.2,
	shear_range = 0.15,
	horizontal_flip = True,
	fill_mode = "nearest")

#------------------Building the model--------------
# loading MobileNetV2 for the fine tuning of our base mode 
# leaving the top layer and adding a few to make it 
# compatible with the expected output
baseModel = MobileNetV2(weights = "imagenet",include_top = False,
	input_tensor = Input(shape = (image_size[0], image_size[1], 3)))

# constructing head of the model to be placed at top of the base model
head_model = baseModel.output
head_model = AveragePooling2D(pool_size=(7,7))(head_model)
head_model = Flatten()(head_model)
head_model = Dense(128, activation="relu")(head_model)
head_model = Dropout(0.4)(head_model)
head_model = Dense(2, activation="softmax")(head_model)

model = Model(inputs = baseModel.input,outputs = head_model)

# looping over base model layers and freezing them 
# so they are not updated during first training process
for layer in baseModel.layers:
	layer.trainable = False

# compiling model
optim=Adam(lr=LR)
model.compile(loss="binary_crossentropy", optimizer=optim,metrics=["accuracy"])

#----------------Training the model on our dataset---------------
History=model.fit(
	aug.flow(train_X,train_Y,batch_size=Batch_Size),
	steps_per_epoch=len(train_X)//Batch_Size,
	validation_data=(valid_X,valid_Y),
	validation_steps=len(valid_X)//Batch_Size,
	epochs=epochs)

model.save("mask_detector.model",save_format="h5")
