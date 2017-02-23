import numpy as np
from scipy import ndimage
import tensorflow as tf
from nn_2_layers import *


BATCH_SIZE = 128
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_LABELS = 10
LAYER1_NODES = 1024
LAYER2_NODES = 256
LAYER3_NODES = 128

PIXEL_DEPTH = 255.0  # Number of levels per pixel.


def load_letter(image_file):
    """Load the data for a single letter label."""
    dataset = np.ndarray(shape=(IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    try:
        image_data = (ndimage.imread(image_file).astype(float) - PIXEL_DEPTH / 2) / PIXEL_DEPTH
        if image_data.shape != (IMAGE_SIZE, IMAGE_SIZE):
            raise Exception('Unexpected image shape: %s' % str(image_data.shape))
        dataset = image_data
    except IOError as e:
        print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    return dataset


def predict():

    # Build the graph
    graph = tf.Graph()
    with graph.as_default():

        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(tf.float32,
                                          shape=(BATCH_SIZE, IMAGE_PIXELS))
        # Variables.
        variables = inference(tf_train_dataset)

        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        saver.restore(sess, './save/nn_2_layer.ckpt')
        # Init handler
        print("Model restored.")
        print('Initialized')
        w1 = sess.run(variables['weights']['hidden_1'])
        print(w1)


predict()
