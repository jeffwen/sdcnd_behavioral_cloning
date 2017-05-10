import cv2
import pandas as pd
import numpy as np
import csv
import platform
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

if platform.system() == "Linux":
    col_names = ["center", "left", "right", "steering", "throttle", "brake", "speed"]
    proj_path = '/home/carnd/sdcnd_behavioral_cloning/'
    folder_path = '/home/carnd/sdcnd_behavioral_cloning/training_video_log/driving_log.csv'
elif platform.system() == "Darwin" and platform.uname()[1] == 'C02RH2F7G8WM':
    folder_path = '/Users/jwen/Python/Projects/sdcnd_behavioral_cloning/training_video_log/driving_log.csv'
    proj_path = '/Users/jwen/Python/Projects/sdcnd_behavioral_cloning/'
elif platform.system() == "Darwin":
    folder_path = '/Users/Jeffwen/Documents/Online Courses/sdcnd_behavioral_cloning/training_video_log/driving_log.csv'
    proj_path = '/Users/Jeffwen/Documents/Online Courses/sdcnd_behavioral_cloning/'
else:
    print("Unknown environment")

    
def read_input(folder_path):

    lines = []

    ## read in the csv file with images and steering data
    with open(folder_path) as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            lines.append(line)

    ## shuffle the observations
    lines_shuffled = shuffle(lines)

    ## split into training and validation data sets
    train_lines, validation_lines = train_test_split(lines_shuffled, test_size = 0.2)

    return train_lines, validation_lines


def generate_data(observations, batch_size=48):

    ## set up generator
    while True:
        for offset in range(0, len(observations), batch_size):
            batch_obs = observations[offset:offset+batch_size]

            center_images = []
            left_images = []
            right_images = []

            steering_angle_center = []
            steering_angle_left = []
            steering_angle_right = []

            steering_correction = 0.25 ## applying correction to left and right steering angles

            ## loop through lines and append images path/ steering data to new lists
            for observation in batch_obs:
                center_image_path = proj_path + 'training_video_log/IMG/'+observation[0].split('/')[-1]
                left_image_path = proj_path + 'training_video_log/IMG/'+observation[1].split('/')[-1]
                right_image_path = proj_path + 'training_video_log/IMG/'+observation[2].split('/')[-1]

                center_images.append(cv2.imread(center_image_path))
                left_images.append(cv2.imread(left_image_path))
                right_images.append(cv2.imread(right_image_path))

                ## append the steering angles and correct for left/right images
                steering_angle_center.append(float(observation[3]))
                steering_angle_left.append(float(observation[3]) + steering_correction)
                steering_angle_right.append(float(observation[3]) - steering_correction)

            images = center_images + left_images + right_images
            steering_angles = steering_angle_center + steering_angle_left + steering_angle_right

            X = np.array(images)
            y = np.array(steering_angles)

            yield shuffle(X, y)
