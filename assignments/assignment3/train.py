import os
import random
import numpy as np
import pandas as pd
from models.settings import *
from keras.callbacks import ModelCheckpoint
from models.model import initialize_model
from generator import generator_superviser 

seed = 20
np.random.seed(seed)

# initialize model
model = initialize_model()
#model.load_weights("model_params/weights-improvement-46-0.65.hdf5")

# number of training examples
NB_TRAIN_SMPL = sum([len(files) for r, d, files in os.walk(TRAIN_DATA_DIR)])

# number of validation examples
NB_VAL_SMPL = sum([len(files) for r, d, files in os.walk(VALIDATION_DATA_DIR)])

# configure checkpoints
filepath=WEIGHTS_DIR + "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# train the model
history = model.fit_generator(
             generator_superviser(TRAIN_DATA_DIR),
             steps_per_epoch= NB_TRAIN_SMPL//BATCH_SIZE,
             epochs=NB_EPOCHS,
             validation_data=generator_superviser(VALIDATION_DATA_DIR),
             validation_steps=NB_VAL_SMPL//BATCH_SIZE,
             callbacks = callbacks_list, workers=1)

# save history
pd.DataFrame(history.history).to_csv("history.csv")

