'''
Train a simple model on this data using 50, 100, 1000 and 5000 training samples.
Hint: you can use the LogisticRegression model from sklearn.linear_model.
'''
from sklearn import linear_model
import numpy as np

def fit_model(number_samples, train_dataset, train_labels):
    idx =np.random.choice(train_labels.shape[0], number_samples, replace=False)
    fit_dataset = train_dataset[idx, :]
    fit_labels = train_labels[idx]
    print('{}:{}'.format('Fit Dataset shape', fit_dataset.shape))
    print('{}:{}'.format('Fit labels shape', fit_labels.shape))
    regr = linear_model.LogisticRegression()
    regr.fit(fit_dataset, fit_labels)

    return regr
