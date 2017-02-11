import numpy as np
import matplotlib.pyplot as plt

from load_data import maybe_download
from load_data import maybe_extract
from curate_data import maybe_pickle

from problem_2 import displaySampleData
from problem_3 import checkDataIsBalanced
from mergeAndPrune import merge_datasets
from mergeAndPrune import randomize
from mergeAndPrune import saveData
from problem_5 import measure_overlap

from predict import predict

# TODO Move this global values somewhere else
image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.
data_root = '/home/jose/WorkingData/ML-DL-Course/' # Change me to store data elsewhere

pickle_filename = 'notMNIST.pickle'
train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)

train_folders = maybe_extract(train_filename)
test_folders = maybe_extract(test_filename)

train_datasets = maybe_pickle(train_folders, 45000)
test_datasets = maybe_pickle(test_folders, 1800)

displaySampleData(train_datasets)

checkDataIsBalanced(train_datasets)

train_size = 200000
valid_size = 10000
test_size = 10000

train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(train_datasets, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)


train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

letter = 'ABCDEFGHIJ'
sample_idx = np.random.randint(0, len(train_dataset))

plt.imshow(train_dataset[sample_idx])
plt.title("Char " + letter[(train_labels[sample_idx])])
plt.show()

saveData(data_root, pickle_filename, train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels)

measure_overlap(train_dataset, valid_dataset, test_dataset)

# train_sanitized, valid_sanitized, test_sanitizes = sanitize(train_dataset, valid_dataset, test_dataset)
# saveSanitizedData(train_sanitized, valid_sanitized, test_sanitizes)
regr = predict(50, train_dataset, train_labels, test_dataset, test_labels)
# The coefficients
print('Coefficients: \n', regr.coef_)
