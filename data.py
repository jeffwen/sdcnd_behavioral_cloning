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

    
## reading input data paths in from the csv
def read_input(folder_path):
    ''' reading in the input data
    Arguments
        folder_path: folder path to the images and csv

    Returns
        train_lines: array of observation file paths and steering angle information from csv for training data
        validation_lines: array of observation file paths and steering angle information from csv for validation data
    '''

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

    return np.array(train_lines), np.array(validation_lines)


## convert color of the input image to YUV as mentioned in nvidia paper and crop image
def preprocess_image(img, color_conversion=cv2.COLOR_BGR2YUV):
    ''' convert color of the input image and crop
    Arguments
        img: individual image array
        color_conversion: color conversion to be performed

    Returns
        cropped_img: cropped and color corrected image array
    '''

    ## convert the color of the image 
    converted_img = cv2.cvtColor(img,color_conversion)

    ## crop the image 
    cropped_img = converted_img[60:140,:,:]

    return cropped_img


## create a relatively uniform distribution of steering angle observations
def distribute_data(observations, min_needed=500, max_needed=750):
    ''' create a relatively uniform distribution of images
    Arguments
        observations: the array of observation data that comes from the read input function
        min_needed: minimum number of observations needed per bin in the histogram of steering angles
        max_needed:: maximum number of observations needed per bin in the histogram of steering angles

    Returns
        observations_output: output of augmented data observations
    '''
    
    observations_output = observations.copy()
    
    ## create histogram to know what needs to be added
    steering_angles = np.asarray(observations_output[:,3], dtype='float')
    num_hist, idx_hist = np.histogram(steering_angles, 20)
    
    to_be_added = np.empty([1,7])
    to_be_deleted = np.empty([1,1])
    
    for i in range(1, len(num_hist)):
        if num_hist[i-1]<min_needed:

            ## find the index where values fall within the range 
            match_idx = np.where((steering_angles>=idx_hist[i-1]) & (steering_angles<idx_hist[i]))[0]

            ## randomly choose up to the minimum needed
            need_to_add = observations_output[np.random.choice(match_idx,min_needed-num_hist[i-1]),:]
            
            to_be_added = np.vstack((to_be_added, need_to_add))

        elif num_hist[i-1]>max_needed:
            
            ## find the index where values fall within the range 
            match_idx = np.where((steering_angles>=idx_hist[i-1]) & (steering_angles<idx_hist[i]))[0]
            
            ## randomly choose up to the minimum needed
            to_be_deleted = np.append(to_be_deleted, np.random.choice(match_idx,num_hist[i-1]-max_needed))

    ## delete the randomly selected observations that are overrepresented and append the underrepresented ones
    observations_output = np.delete(observations_output, to_be_deleted, 0)
    observations_output = np.vstack((observations_output, to_be_added[1:,:]))
    
    return observations_output


## data generator
def generate_data(observations, batch_size=128):
    ''' data generator in batches to be fed into the Keras fit_generator object
    Arguments
        observations: the array of observation data that is to be split into batches and read into image arrays
        batch_size: batches of images to be fed to Keras model

    Returns
        X: image array in batches as a list
        y: steering angle list 
    '''

    ## applying correction to left and right steering angles
    steering_correction = 0.2 
    
    ## set up generator
    while True:
        for offset in range(0, len(observations), batch_size):
            batch_obs = shuffle(observations[offset:offset+batch_size])

            center_images = []
            left_images = []
            right_images = []

            steering_angle_center = []
            steering_angle_left = []
            steering_angle_right = []

            ## loop through lines and append images path/ steering data to new lists
            for observation in batch_obs:

                center_image_path = proj_path + 'training_video_log/IMG/'+observation[0].split('/')[-1]
                left_image_path = proj_path + 'training_video_log/IMG/'+observation[1].split('/')[-1]
                right_image_path = proj_path + 'training_video_log/IMG/'+observation[2].split('/')[-1]

                center_images.append(preprocess_image(cv2.imread(center_image_path)))
                steering_angle_center.append(float(observation[3]))

                left_images.append(preprocess_image(cv2.imread(left_image_path)))
                right_images.append(preprocess_image(cv2.imread(right_image_path)))

                ## append the steering angles and correct for left/right images
                steering_angle_left.append(float(observation[3]) + steering_correction)
                steering_angle_right.append(float(observation[3]) - steering_correction)

            images = center_images + left_images + right_images
            steering_angles = steering_angle_center + steering_angle_left + steering_angle_right

            X = np.array(images)
            y = np.array(steering_angles)

            yield shuffle(X, y)
