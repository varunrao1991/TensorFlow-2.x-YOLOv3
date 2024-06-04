#================================================================
#
#   File name   : detect_mnist.py
#   Author      : PyLessons
#   Created date: 2020-08-12
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : mnist object detection example
#
#================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import random
import time
import tensorflow as tf
from yolov3.yolov4 import Create_Yolo
from yolov3.utils import detect_image
from yolov3.configs import *

while True:
    label_txt = "mnist/mnist_test.txt"
    labels = open(label_txt).readlines()
    ID = random.randint(0, len(labels)-1)
    image_info = labels[ID].split()

    image_path = image_info[0]

    yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
    yolo.load_weights(f"./checkpoints/{TRAIN_MODEL_NAME}") # use keras weights

    detect_image(yolo, image_path, "mnist_test.jpg", input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))

label_txt = "mnist/mnist_test.txt"
labels = open(label_txt).readlines()
index = 1
yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
yolo.load_weights(f"./checkpoints/{TRAIN_MODEL_NAME}") # use keras weights

while True:
    ID = index % len(labels)
    index = index + 1
    image_info = labels[ID].split()

    image_path = image_info[0]
    image = detect_image(yolo, image_path, "objects_test.jpg", input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0), iou_threshold = 0.1)

    cv2.imshow("predicted image", image)
    cv2.waitKey(500)