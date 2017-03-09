import numpy as np
import os
from six.moves import cPickle as pickle

import fully_connected_nn_2

from config import open_config_if_exists
import json


# Load the local config file
with open_config_if_exists('../config/local_config.json') as (config_file, err):
    if err:
        print('Local Config file is not defined. Expected at ./config/local_config.json')
    else:
        local_config_data = json.load(config_file)

with open_config_if_exists('nn2_shared_config.json') as (config_file, err):
    if err:
        print('Shared Config file is not defined. Expected at ./config/local_config.json')
    else:
        shared_config_data = json.load(config_file)


data_root = local_config_data['dataRoot']

NUM_LABELS = shared_config_data['outputSize']
IMAGE_PIXELS = shared_config_data['imageSize'] * shared_config_data['imageSize']

TRAINING_RATE = 0.5
reg_beta = 1e-3
dropout = 0.0
overfitting = False


def load_data(data_dir):
    pickle_file = os.path.join(data_dir, 'notMNIST.pickle')

    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  # hint to help gc free up memory
        return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, IMAGE_PIXELS)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(NUM_LABELS) == labels[:, None]).astype(np.float32)
    return dataset, labels


train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = load_data(data_root)

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


fully_connected_nn_2.run_training(shared_config_data, TRAINING_RATE, train_dataset, train_labels, valid_dataset,
                                  valid_labels, test_dataset, test_labels, reg_beta)
