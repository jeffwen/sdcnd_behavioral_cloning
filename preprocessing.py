import cv2
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm

col_names = ["center","left","right","steering","throttle","brake","speed"]
folder_path = '/Users/Jeffwen/Documents/Online Courses/sdcnd_behavioral_cloning/training_video_log/driving_log.csv'

def read_input(folder_path):

    lines = []

    ## read in the csv file with images and steering data
    with open(folder_path) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            lines.append(line)

    center_images = []
    left_images = []
    right_images = []

    steering_angle_center = []
    steering_angle_left = []
    steering_angle_right = []

    steering_correction = 0.25 ## applying correction to left and right steering angles

    ## loop through lines and append images path/ steering data to new lists
    for line in tqdm(lines):
        center_images.append(cv2.imread(line[0]))
        left_images.append(cv2.imread(line[1]))
        right_images.append(cv2.imread(line[2]))

        ## append the steering angles and correct for left/right images
        steering_angle_center.append(float(line[3]))
        steering_angle_left.append(float(line[3]) + steering_correction)
        steering_angle_right.append(float(line[3]) - steering_correction)

    images = center_images + left_images + right_images
    steering_angles = steering_angle_center + steering_angle_left + steering_angle_right

    return np.array(images), np.array(steering_angles)
        
test = []
with open(folder_path) as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        test.append(line)
