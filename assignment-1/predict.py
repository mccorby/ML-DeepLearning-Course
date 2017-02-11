'''
Train a simple model on this data using 50, 100, 1000 and 5000 training samples.
Hint: you can use the LogisticRegression model from sklearn.linear_model.
'''
from sklearn import linear_model
import numpy as np

def predict(number_samples, train_dataset, train_labels, test_dataset, test_labels):
    train_dataset_size = len(train_dataset)
    # Pure functions don't do this!
    print('{} {}'.format('Train Dataset Size', train_dataset_size))
    reshaped_dataset = train_dataset.reshape(train_dataset_size, -1)
    idx = np.random.randint(number_samples)
    predict_dataset = train_dataset[idx, :]
    regr = linear_model.LogisticRegression()
    regr.fit(reshaped_dataset, train_labels)
    return regr
