import numpy as np
import data
from matplotlib import pyplot as plt
import seaborn as sns
import platform
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.callbacks import ModelCheckpoint


if platform.system() == "Linux":
    col_names = ["center","left","right","steering","throttle","brake","speed"]
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

    
## read in the training data and split into train and validation    
train_observations, validation_observations = data.read_input(folder_path)

## create data generators
train_generator = data.generate_data(train_observations)
validation_generator = data.generate_data(validation_observations)


## build Keras model
model = Sequential()

## cropping images
model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160,320,3)))

## batch normlization
model.add(BatchNormalization())

## Convolutional layers
model.add(Conv2D(24, 5, 5, subsample=(2,2), activation='elu'))
model.add(Conv2D(36, 5, 5, subsample=(2,2), activation='elu'))
model.add(Conv2D(48, 5, 5, subsample=(2,2), activation='elu'))
model.add(Conv2D(64, 3, 3, activation='elu'))
model.add(Conv2D(64, 3, 3, activation='elu'))

model.add(Dropout(0.5))
## fully connected layers
model.add(Flatten())
# model.add(Dense(1164, activation='elu'))
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))

## compile
model.compile(loss='mse', optimizer='adam')

## load and continue training the pre-trained model on more data
# model = load_model(proj_path + "model.h5")

# checkpoint and save best model
model_path = "model.h5"
checkpoint = ModelCheckpoint(model_path, verbose=0, save_best_only=True)
callbacks_list = [checkpoint]

## fit the model
model.fit_generator(train_generator, samples_per_epoch=len(train_observations), validation_data=validation_generator,
                    nb_val_samples=len(validation_observations), nb_epoch=5, callbacks=callbacks_list)

