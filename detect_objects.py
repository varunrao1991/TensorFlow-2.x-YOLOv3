#================================================================
#
#   File name   : detect_objects.py
#   Author      : PyLessons
#   Created date: 2020-08-12
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : objects object detection example
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

label_txt = "objects/objects_test.txt"
labels = open(label_txt).readlines()
index = 18
yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
yolo.load_weights(f"./checkpoints/{TRAIN_MODEL_NAME}") # use keras weights

while True:
    #index = (index + 1) % len(labels)
    image_info = labels[index].split()

    image_path = image_info[0]
    image = detect_image(yolo, image_path, "", input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0), iou_threshold = 0.1)
    print(index)
    cv2.imshow("predicted image", image)
    cv2.waitKey(500)