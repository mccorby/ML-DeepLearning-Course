'''
Train a simple model on this data using 50, 100, 1000 and 5000 training samples.
Hint: you can use the LogisticRegression model from sklearn.linear_model.
'''
impot sklearn.linear_model

def predict(number_samples, train_dataset, train_labels, test_dataset, test_labels):
    idx = np.random.randint(number_samples)
    predict_dataset = train_dataset[idx, :]
    regr = linear_model.LogisticRegression()
    regr.fit(train_dataset, train_labels)
    return regr
