'''
Problem 5
By construction, this dataset might contain a lot of overlapping samples,
including training data that's also contained in the validation and test set!
Overlap between training and test can skew the results if you expect to use your
model in an environment where there is never an overlap, but are actually ok
if you expect to see training samples recur when you use it.
Measure how much overlap there is between training, validation and test samples.
Optional questions:
What about near duplicates between datasets? (images that are almost identical)
Create a sanitized validation and test set, and compare your accuracy on those in subsequent assignments.
'''

from hashlib import md5
import numpy as np

def getHashOfSet(aSet):
    return map(lambda x: md5(x).hexdigest(), aSet)

def measure_overlap(train_dataset, valid_dataset, test_dataset):
    # Using md5 hashing
    train_set_hash = set(getHashOfSet(train_dataset))
    valid_set_hash = set(getHashOfSet(valid_dataset))
    test_set_hash = set(getHashOfSet(test_dataset))

    print('{} {}'.format('Overlap train/validation', len(valid_set_hash - train_set_hash)))
    print('{} {}'.format('Overlap train/test %d', len(test_set_hash - train_set_hash)))

def merge_features_labels(features, labels):
    # Add the column with labels to the features
    pass

def sanitize(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels):
    # Merge dataset with labels
    # Get hash of each element in each set

    all_data_set_hash = set(getHashOfSet(train_dataset))
    valid_set_hash = set(getHashOfSet(valid_dataset))
    test_set_hash = set(getHashOfSet(test_dataset))
    all_data = np.lexsort()

    # Apply lex-sorting
    sorted_idx = np.lexsort(all_data)
