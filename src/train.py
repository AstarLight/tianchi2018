"""
Author: Lijunshi
Date: 2018-9-30
"""

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
 
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
import sys
import logging
from net.mynet import MyNet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logging.basicConfig(level=logging.DEBUG)

label_dict = {"正常":0, "不导电":1, "擦花":2, "横条压凹":3, "桔皮":4, "漏底":5, "碰伤":6, "起坑":7, "凸粉":8,
        "涂层开裂":9, "脏点":10, "其他":11}

def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-dtest", "--dataset_test", required=False,
        help="path to input dataset_test")
    ap.add_argument("-dtrain", "--dataset_train", required=True,
        help="path to input dataset_train")
    ap.add_argument("-m", "--model", required=True,
        help="path to output model")
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
        help="path to output accuracy/loss plot")
    args = vars(ap.parse_args()) 
    return args


# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS = 120
INIT_LR = 1e-3
BS = 64
CLASS_NUM = 12
norm_size = 128  # image size: 256*256
test_rate = 0.2

def load_data(path):
    print("[INFO] loading images...")
    test_data = []
    test_labels = []
    train_data = []
    train_labels = []
    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(path)))
    random.seed(42)
    random.shuffle(imagePaths)
    test_num = int(test_rate * len(imagePaths))
    logging.info("total image num is %s", len(imagePaths))
    logging.info("test data num is %s", test_num)
    count = 0
    # loop over the input images
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        logging.debug("image path is %s", imagePath)
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (norm_size, norm_size))
        image = img_to_array(image)
        # data.append(image)

        # extract the class label from the image path and update the
        # labels list
        label_name = imagePath.split(os.path.sep)[-1].split(".")[0].split("2018")[0]
        logging.debug("label name is %s", label_name)
        try:
            label = int(label_dict[label_name])
        except:
            label = 11
        logging.debug("label is %s", label)
        if(count < test_num):
            test_data.append(image)
            test_labels.append(label)
        else:
            train_data.append(image)
            train_labels.append(label)
        count += 1;
    
    # scale the raw pixel intensities to the range [0, 1]
    test_data = np.array(test_data, dtype="float") / 255.0
    train_data = np.array(train_data, dtype="float") / 255.0
    test_labels = np.array(test_labels)
    trian_labels = np.array(train_labels)

    # convert the labels from integers to vectors
    test_labels = to_categorical(test_labels, num_classes=CLASS_NUM)                         
    train_labels = to_categorical(train_labels, num_classes=CLASS_NUM)
    return test_data, test_labels, train_data, train_labels
    

def train(aug, trainX, trainY, testX, testY, args):
    # initialize the model
    print("[INFO] compiling model...")
    model = MyNet.build(width=norm_size, height=norm_size, depth=3, classes=CLASS_NUM)
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # train the network
    print("[INFO] training network...")
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
                            validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
                            epochs=EPOCHS, verbose=1)

    # save the model to disk
    print("[INFO] serializing network...")
    model.save(args["model"])
    
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on tianchi2018 classifier")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])
    

# python train.py --dataset_train /home/ljs/tianchi2018/data_all1 --model lenet_100.model
if __name__=='__main__':
    args = args_parse()
    train_file_path = args["dataset_train"]
    # test_file_path = args["dataset_test"]
    testX, testY, trainX, trainY = load_data(train_file_path)
    # testX,testY = load_data(test_file_path)
    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")
    train(aug, trainX, trainY, testX, testY, args)
