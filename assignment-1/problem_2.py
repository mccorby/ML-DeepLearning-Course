import numpy as np
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle

'''
Problem 2
Let's verify that the data still looks good. Displaying a sample of the labels and images from the ndarray.
Hint: you can use matplotlib.pyplot.
'''

def displaySampleData(train_datasets):
    print(train_datasets[0])
    with open(train_datasets[0], 'rb') as f:
        letter_set = pickle.load(f)
    plt.imshow(letter_set[0])
    plt.title("Char a")
