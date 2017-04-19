# train data directory
TRAIN_DATA_DIR = 'data/train/'

# validation data directory
VALIDATION_DATA_DIR = 'data/validation/'

# test data directory
TEST_DATA_DIR = 'data/test/'

# weights directory
WEIGHTS_DIR = 'model_params/'

# number of epochs
NB_EPOCHS = 100

# batch size
BATCH_SIZE = 10

# learning rate
LEARNING_RATE = 1e-4

# number of output classes
NB_CLASSES = 6

# number of features per time step
NB_FEATURES = 9

# number of time steps
NB_TIMESTEPS = 5

# list of activities
ACTIVITIES_LIST = ['a0'+'{0:1s}'.format(str(i+1)) if i+1 <= 9
                           else 'a'+'{0:2s}'.format(str(i+1)) for i in range(NB_CLASSES)]

