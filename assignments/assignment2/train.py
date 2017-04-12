import os
import numpy as np
import pandas as pd
from models.settings import *
from models.utils import quadratic_weighted_kappa, get_class_weights 
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from models.model import initialize_model

#######################################################
#                                                     #
#             Simple ConvNet model                    #
#                                                     #
#######################################################

seed = 9
np.random.seed(seed)

# Initialize generators
train_datagen = ImageDataGenerator()
test_datagen  = ImageDataGenerator()

# Generator data flows
train_generator = train_datagen.flow_from_directory(
                                           TRAIN_DATA_DIR,
                                           class_mode='categorical', shuffle=True,
                                           target_size=(IMG_WIDTH, IMG_HEIGHT),
                                           batch_size=BATCH_SIZE)

validation_generator = test_datagen.flow_from_directory(
                                           VALIDATION_DATA_DIR,
                                           class_mode='categorical', shuffle=True,
                                           target_size=(IMG_WIDTH, IMG_HEIGHT),
                                           batch_size=BATCH_SIZE)

# Initialize the model
model = initialize_model()
#model.load_weights("model_params/weights-improvement-46-0.65.hdf5")

# get the number of training examples
NB_TRAIN_SMPL = sum([len(files) for r, d, files in os.walk(TRAIN_DATA_DIR)])

# get the number of validation examples
NB_VAL_SMPL = sum([len(files) for r, d, files in os.walk(VALIDATION_DATA_DIR)])

# get class weights for dealing with class imbalances
correct_train_labels = pd.read_csv('trainLabels.csv')
labels_list = correct_train_labels['level'].values.tolist()
class_weights = get_class_weights(labels_list)

# configure checkpoints
filepath=WEIGHTS_DIR + "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
model_checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [ model_checkpoint]

history = model.fit_generator(
             train_generator,
             steps_per_epoch= NB_TRAIN_SMPL//BATCH_SIZE,
             epochs=NB_EPOCHS,
             validation_data=validation_generator,
             validation_steps=NB_VAL_SMPL//BATCH_SIZE,
             class_weight=class_weights,
             callbacks = callbacks_list)

# save history
pd.DataFrame(history.history).to_csv("history.csv")
