from models.settings import *
import numpy as np
import pandas as pd
import os

def data_generator(input_dir):

    # create a list of all training files and correct labels
    input_files = []
    correct_labels = []
    for activity in ACTIVITIES_LIST:
        for _, _, filenames in os.walk(input_dir+activity+"/"):
            for filename in filenames:
                input_files.append(str(input_dir+activity+"/"+filename))
                correct_labels.append(activity)

    assert len(input_files)==len(correct_labels)
    input_files = np.array(input_files)
    correct_labels = np.array(correct_labels)

    # shuffle files and indexes (NOTE: we do not directly apply shuffle to matrices because they should
    # be shuffled synchronically)'
    shuffle_idx = np.arange(len(correct_labels))
    np.random.shuffle(shuffle_idx)
    input_files = input_files[shuffle_idx]
    correct_labels = correct_labels[shuffle_idx]

    # get one-hot vectors of correct labels
    def one_hot_transform(array, voc=ACTIVITIES_LIST):
        int_indexes = map(voc.index, array)
        return map(lambda idx: map(int, [ix == idx for ix in range(NB_CLASSES)]), int_indexes)

    encoded_correct_labels = np.array(one_hot_transform(correct_labels))

    # iterate over the data
    while True:

        if input_files.shape[0] < BATCH_SIZE:
            break

        def get_data_from_file(file):
            data = pd.read_csv(file, index_col=0)
            return np.array(data.values.tolist())

        input_files_batch, input_files = input_files[:BATCH_SIZE], input_files[BATCH_SIZE:]
        correct_labels_batch, encoded_correct_labels = encoded_correct_labels[:BATCH_SIZE], encoded_correct_labels[BATCH_SIZE:]

        input_data_batch = map(get_data_from_file, input_files_batch)
        input_data_batch = np.array(input_data_batch)

        yield input_data_batch, correct_labels_batch

def generator_superviser(input_dir):
    generator = data_generator(input_dir)
    while True:
        try:
            X, y = generator.next()
        except:
            # re-initialize the generator if it runs out of data
            generator = data_generator(input_dir)
            X, y = generator.next()
        yield X, y

