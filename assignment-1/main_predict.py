import os
from six.moves import cPickle as pickle
from sklearn.externals import joblib
import matplotlib.pyplot as plt

# Load data
letters = 'ABCDEFGHIJ'
data_root = '/home/jose/WorkingData/ML-DL-Course/'
data_file = 'notMNIST.pickle'
file_path = os.path.join(data_root, data_file)
if not os.path.exists(file_path):
    raise Exception('File ' + file_path + ' does not exist. Run main.py to generate it')
with open(file_path, 'rb') as f:
    dataset = pickle.load(f)

print('{}:{}'.format('Entire dataset shape', len(dataset)))
valid_dataset = dataset['valid_dataset']
valid_labels = dataset['valid_labels']

print('{} {}/{}'.format('Size Validation Datasete', valid_dataset.shape, valid_labels.shape))

# Load model
model_filename = './models/model1000.pkl'
model = joblib.load(model_filename)

# Predict using validation set. Ideally we should be using any other image
sample = valid_dataset[12, :]
print(sample.shape)
plt.imshow(sample)
plt.show()
# Reshape your data X.reshape(1, -1) if it contains a single sample.
sample_shaped = sample.reshape(1, -1)
result = model.predict(sample_shaped)
print(result)
print(letters[result[0]])
