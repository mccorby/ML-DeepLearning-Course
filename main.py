import fully_connected_logreg
import logistic_regression
import fully_connected_relu
import os
from six.moves import cPickle as pickle
import numpy as np


def load_data(data_root):
    pickle_file = os.path.join(data_root, 'notMNIST.pickle')

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
    dataset = dataset.reshape((-1, logistic_regression.IMAGE_PIXELS)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(logistic_regression.NUM_LABELS) == labels[:, None]).astype(np.float32)
    return dataset, labels


TRAINING_RATE = 0.5

data_root = '/Users/jco59/ML/WorkingData/DL-Course/assignment1'
train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = load_data(data_root)

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# fully_connected_logreg.run_training(TRAINING_RATE, train_dataset, train_labels, valid_dataset, valid_labels,
#                                    test_dataset, test_labels)

log_report_dir = os.path.join(data_root, '../reports')

fully_connected_relu.run_training(TRAINING_RATE, train_dataset, train_labels, valid_dataset, valid_labels,
                                  test_dataset, test_labels, log_report_dir)
