from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers.recurrent import LSTM
from settings import *

def initialize_model():

    model = Sequential()
    model.add(LSTM(30, input_shape=(NB_TIMESTEPS, NB_FEATURES)))
    model.add(Dense(NB_CLASSES))
    model.add(Activation('softmax'))

    opt = Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics=['accuracy'])

    print model.summary()

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    return model

