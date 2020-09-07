"""
Image Inference:
Takes the path to the unseen inference images directory and outputs images with bounding
boxes around detections.
"""


# import modulesimport os
import os
import cv2
import glob
import time
import keras
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras_retinanet import models
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.visualization import  draw_caption
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image


# construct argument parser
ap = argparse.ArgumentParser()

ap.add_argument("-i", "--input_dir", required=True, help = "dir path to inference images")
ap.add_argument("-t", "--threshold", default=0.5, type = float, help = "threshold for filtering weak detections")
ap.add_argument("-m", "--model", required=True, help = "path to trained/converted model")
ap.add_argument("-o", "--output_dir", required=True, help = "path to output directory")

args = vars(ap.parse_args())


# create variables for arguments
input_path = args["input_dir"]
output_path = args["output_dir"]
THRES_SCORE = args["threshold"]
model = models.load_model(args["model"], backbone_name='resnet50')
inference_images = [os.path.join(input_path, file) for file in glob.glob(input_path + '*.jpg')]


def draw_box(image, box, color, thickness=15):
    """ Draws a box on an image with a given color.

    # Arguments
        image     : The image to draw on.
        box       : A list of 4 elements (x1, y1, x2, y2).
        color     : The color of the box.
        thickness : The thickness of the lines to draw a box with.
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)

# loop over inference images
for (i, img_path) in enumerate(inference_images):

    print("[INFO] saving image {} of {}".format(i+1, len(inference_images)))

    #load image
    image = read_image_bgr(img_path)

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    (image, scale) = resize_image(image)
    image = np.expand_dims(image, axis=0)

    # process image and correct for image scale
    start = time.time()
    (boxes, scores, labels) = model.predict_on_batch(image)
    print("processing time: ", time.time() - start)
    boxes /= scale
    # visualize detections
    for (box, score, label) in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < THRES_SCORE:
            continue

        color = label_color(label)
        b = box.astype(int)

        # draw bounding box
        draw_box(draw, b, color=(230,239,36))
        plt.figure(figsize=(20, 20))
        plt.axis('off')
        plt.imshow(draw)
      
        # save in output path
        plt.savefig(os.path.join(output_path, "bb_" + os.path.basename(img_path)))
        

print("[FINAL] task completed!")

    #     caption = "{} {:.3f}".format(labels_to_names[label], score)
    #     draw_caption(draw, b, caption)
