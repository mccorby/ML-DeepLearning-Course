'''
We expect the data to be balanced across classes. Verify that.
'''
from six.moves import cPickle as pickle

allLetters = 'ABCDEFGHIJ';

def letter(i):
    return allLetters[i]

def checkDataIsBalanced(train_datasets):
    for i in range(0, len(allLetters)):
        with open(train_datasets[i], 'rb') as f:
            letter_set = pickle.load(f)
        print('Size of the data for class %s: %d', allLetters[i], letter_set.size)
