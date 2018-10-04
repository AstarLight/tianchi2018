"""
Author: Lijunshi
Date: 2018-9-30
"""

# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K
from keras.layers.core import Dropout


class MyNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)
        # if we are using "channels last", update the input shape
        if K.image_data_format() == "channels_first":  # for tensorflow
            inputShape = (depth, height, width)
        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(64, (3, 3), padding="same", input_shape=inputShape, strides=(1, 1)))
        model.add(Activation("relu"))
        model.add(Conv2D(64, (3, 3), padding="same", input_shape=inputShape, strides=(1, 1)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(128, (3, 3), padding="same", strides=(1, 1)))
        model.add(Activation("relu"))
        model.add(Conv2D(128, (3, 3), padding="same", strides=(1, 1)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(256, (3, 3), padding="same", strides=(1, 1)))
        model.add(Activation("relu"))
        model.add(Conv2D(256, (3, 3), padding="same", strides=(1, 1)))
        model.add(Activation("relu"))
        model.add(Conv2D(256, (3, 3), padding="same", strides=(1, 1)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(256, (3, 3), padding="same", strides=(1, 1)))
        model.add(Activation("relu"))
        model.add(Conv2D(256, (3, 3), padding="same", strides=(1, 1)))
        model.add(Activation("relu"))
        model.add(Conv2D(256, (3, 3), padding="same", strides=(1, 1)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model
