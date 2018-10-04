"""
Author: Lijunshi
Date: 2018-9-30
"""

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import csv
import imutils
import cv2
import os
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.basicConfig(level=logging.DEBUG)

norm_size = 128

def args_parse():
# construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
        help="path to trained model model")
    ap.add_argument("-d", "--dtest", required=True,
        help="the dir of test images")
    ap.add_argument("-s", "--show", action="store_true",
        help="show predict image",default=False)
    args = vars(ap.parse_args())    
    return args

    
def predict(args):
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model(args["model"])
    
    data_dir = args["dtest"]
    image_list = os.listdir(data_dir)
    out = open("./submit.csv", "w+", newline="")
    csv_writer = csv.writer(out, dialect="excel")
    for i in range(len(image_list)):
        # load the image
        image_name = str(i)+".jpg"
        full_path = os.path.join(data_dir, image_name)
        image = cv2.imread(full_path)

        # pre-process the image for classification
        image = cv2.resize(image, (norm_size, norm_size))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # classify the input image
        result = model.predict(image)[0]
        proba = np.max(result)
        label = str(np.where(result == proba)[0][0])
        logging.debug("%s predict as %s, probability is %s", full_path, label, proba)
        if label == "0":
            final_label = "norm"
        else:
            final_label = "defect" + label
        csv_data = [image_name, final_label]
        csv_writer.writerow(csv_data)

"""
        # label = str(np.where(result==proba)[0])
        # label = "{}: {:.2f}%".format(label, proba * 100)
        # print(label)
        
        if args['show']:   
            # draw the label on the image
            output = imutils.resize(orig, width=400)
            cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)       
            # show the output image
            cv2.imshow("Output", output)
            cv2.waitKey(0)
"""

# python predict.py --model tianchi.model -dtest /home/ljs/tianchi2018/dataset/guangdong_round1_test_a_20180916
if __name__ == '__main__':
    args = args_parse()
    predict(args)
