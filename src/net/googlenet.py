"""
Author: Cong
Date: 2018-10-07
"""
from keras.layers import *
from keras.models import Model
from keras import optimizers
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator



class GoogleNet:
    
    # Inception module - main building block
    def inception_module(X, filter_sizes):

        # 1x1 covolution
        conv_1x1 = Conv2D(filter_sizes[0], kernel_size=1, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.0002))(X)

        # Bottleneck layer and 3x3 convolution
        conv_3x3 = Conv2D(filter_sizes[1], kernel_size=1, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.0002))(X)
        conv_3x3 = Conv2D(filter_sizes[2], kernel_size=3, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.0002))(conv_3x3)

        # Bottleneck layer and 5x5 convolution
        conv_5x5 = Conv2D(filter_sizes[3], kernel_size=1, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.0002))(X)
        conv_5x5 = Conv2D(filter_sizes[4], kernel_size=5, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.0002))(conv_5x5)

        # Max pooling and bottleneck layer
        max_pool = MaxPooling2D(pool_size=3, strides=1, padding='same')(X)
        max_pool = Conv2D(filter_sizes[5], kernel_size=1, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.0002))(max_pool)

        # Concatenate all tensors to 1 tensor
        X = concatenate([conv_1x1, conv_3x3, conv_5x5, max_pool], axis=3)

        return X
    
    # Auxiliary classifier - for predictions in a middle stage
    def aux_classifier(X, classes):

        # Average pooling, fc, dropout, fc
        X = AveragePooling2D(pool_size=3, strides=2, padding='same')(X)
        X = Conv2D(filters=128, kernel_size=1, strides=1, padding='valid', activation='relu', kernel_regularizer=l2(0.0002))(X)
        X = Flatten()(X)
        X = Dense(1024, activation='relu', kernel_regularizer=l2(0.0002))(X)
        X = Dropout(0.7)(X)
        X = Dense(classes, activation='softmax', kernel_regularizer=l2(0.0002))(X)

        return X

    # Full model
    def build(img_rows, img_cols, color_type=1, num_classes=None):
        classes = num_classes
        # Define the input
        X_input = Input(shape=(img_rows, img_cols, color_type), name='data')

        # Stage 1 - layers before inception modules
        X = Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.0002))(X_input)
        X = MaxPooling2D(pool_size=3, strides=2, padding='same')(X)
        X = BatchNormalization(axis=3)(X)
        X = Conv2D(filters=64, kernel_size=1, strides=1, padding='valid', activation='relu', kernel_regularizer=l2(0.0002))(X)
        X = Conv2D(filters=192, kernel_size=3, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.0002))(X)
        X = BatchNormalization(axis=3)(X)
        X = MaxPooling2D(pool_size=3, strides=1, padding='same')(X)

        # Stage 2 - 2 inception modules and max pooling
        X = GoogleNet.inception_module(X, filter_sizes=[64, 96, 128, 16, 32, 32])
        X = GoogleNet.inception_module(X, filter_sizes=[128, 128, 192, 32, 96, 64])
        X = MaxPooling2D(pool_size=3, strides=2, padding='same')(X)

        # Stage 3 - 5 inception modules and max pooling
        X = GoogleNet.inception_module(X, filter_sizes=[192, 96, 208, 16, 48, 64])
        aux_output_1 = GoogleNet.aux_classifier(X, classes)   # Auxiliary classifier
        X = GoogleNet.inception_module(X, filter_sizes=[160, 112, 225, 24, 64, 64])
        X = GoogleNet.inception_module(X, filter_sizes=[128, 128, 256, 24, 64, 64])
        X = GoogleNet.inception_module(X, filter_sizes=[112, 144, 288, 32, 64, 64])
        aux_output_2 = GoogleNet.aux_classifier(X, classes) # Auxiliary classifier
        X = GoogleNet.inception_module(X, filter_sizes=[256, 160, 320, 32, 128, 128])
        X = MaxPooling2D(pool_size=3, strides=2, padding='same')(X)

        # Stage 4 - 2 inception modules and average pooling
        X = GoogleNet.inception_module(X, filter_sizes=[256, 160, 320, 32, 128, 128])
        X = GoogleNet.inception_module(X, filter_sizes=[384, 192, 384, 48, 128, 128])
        X = AveragePooling2D(pool_size=4, strides=1, padding='valid')(X)

        # Stage 5 - dropout, linear fc, softmax fc
        X = Flatten()(X)
        X = Dropout(0.4)(X)
        X_output = Dense(classes, activation='softmax', kernel_regularizer=l2(0.0002))(X)

        # Create model - combine main classifier with auxiliary classifiers
        #model = Model(inputs=X_input, outputs=[X_output, aux_output_1, aux_output_2])
        model = Model(inputs=X_input, outputs=X_output)

        return model
