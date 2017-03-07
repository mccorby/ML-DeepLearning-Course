from convnet import reformat
from convnet_network import *
from load_data import load_data

# TODO data_root from config file
data_root = '/Users/jco59/ML/WorkingData/DL-Course/assignment1'
train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = load_data(data_root)

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


run_training(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels)
