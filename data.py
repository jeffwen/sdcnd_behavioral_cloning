import cv2
import pandas as pd
import numpy as np
import csv
import random
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
    train_lines, validation_lines = train_test_split(lines_shuffled, test_size=0.2)

    return train_lines, validation_lines

## convert color of the input image to YUV as mentioned in nvidia paper
def preprocess_image(img, color_conversion=cv2.COLOR_BGR2YUV):
    converted_img = cv2.cvtColor(img,color_conversion)
    cropped_img = converted_img[60:140,:,:]

    return cropped_img


def generate_data(observations, batch_size=128):

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

            steering_correction = 0.2 ## applying correction to left and right steering angles

            ## loop through lines and append images path/ steering data to new lists
            for observation in batch_obs:

                center_image_path = proj_path + 'training_video_log/IMG/'+observation[0].split('/')[-1]
                left_image_path = proj_path + 'training_video_log/IMG/'+observation[1].split('/')[-1]
                right_image_path = proj_path + 'training_video_log/IMG/'+observation[2].split('/')[-1]

                if np.abs(float(observation[3])) < 0.3:
                    if random.random() > 0.8:
                        center_images.append(preprocess_image(cv2.imread(center_image_path)))
                        steering_angle_center.append(float(observation[3]))
                        
                if np.abs(float(observation[3])) >= 0.3:
                    if random.random() > 0.5:
                        left_images.append(preprocess_image(cv2.imread(left_image_path)))
                        right_images.append(preprocess_image(cv2.imread(right_image_path)))

                        ## append the steering angles and correct for left/right images
                        steering_angle_left.append(float(observation[3]) + steering_correction)
                        steering_angle_right.append(float(observation[3]) - steering_correction)

                        ## flip images
                        right_images.append(preprocess_image(cv2.flip(cv2.imread(left_image_path), 1)))
                        left_images.append(preprocess_image(cv2.flip(cv2.imread(right_image_path), 1)))

                        ## append the steering angles and correct for left/right images
                        steering_angle_left.append(float(observation[3]) + steering_correction)
                        steering_angle_right.append(float(observation[3]) - steering_correction)

                if np.abs(float(observation[3])) >= 0.5:
                        left_images.append(preprocess_image(cv2.imread(left_image_path)))
                        left_images.append(preprocess_image(cv2.imread(left_image_path)))

                        ## append the steering angles and correct for left/right images
                        steering_angle_left.append(float(observation[3]) + steering_correction)
                        steering_angle_right.append(float(observation[3]) - steering_correction)


                        ## flip images
                        right_images.append(preprocess_image(cv2.flip(cv2.imread(left_image_path), 1)))
                        left_images.append(preprocess_image(cv2.flip(cv2.imread(right_image_path), 1)))


                        ## append the steering angles and correct for left/right images
                        steering_angle_left.append(float(observation[3])*-1 + steering_correction)
                        steering_angle_right.append(float(observation[3])*-1 - steering_correction)

            images = center_images + left_images + right_images
            steering_angles = steering_angle_center + steering_angle_left + steering_angle_right

            X = np.array(images)
            y = np.array(steering_angles)

            yield shuffle(X, y)
