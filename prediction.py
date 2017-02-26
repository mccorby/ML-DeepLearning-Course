import numpy as np
from scipy import ndimage
import tensorflow as tf
from tensorflow.core.framework import graph_pb2

from nn_2_layers import *

BATCH_SIZE = 128
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_LABELS = 10
LAYER1_NODES = 1024
LAYER2_NODES = 256
LAYER3_NODES = 128

PIXEL_DEPTH = 255.0  # Number of levels per pixel.
LETTERS = 'ABCDEFGHIJ'


def load_letter(image_file):
    """Load the data for a single letter label."""
    dataset = np.ndarray(shape=(IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    try:
        # image_data = (ndimage.imread(image_file).astype(float) - PIXEL_DEPTH / 2) / PIXEL_DEPTH
        image_data = ndimage.imread(image_file).astype(float)
        dataset = image_data[:, [0, 1]]
        print(dataset.shape)
        if image_data.shape != (IMAGE_SIZE, IMAGE_SIZE):
            raise Exception('Unexpected image shape: %s' % str(image_data.shape))
        dataset = image_data
    except IOError as e:
        print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    return dataset


def predict(dataset):
    # Build the graph
    graph = tf.Graph()
    with graph.as_default():
        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_dataset = tf.placeholder(tf.float32, shape=(1, IMAGE_PIXELS))
        # Variables.
        variables = inference(tf_dataset)

        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        saver.restore(sess, './save/nn_2_layer.ckpt')

        # Init handler
        print("Model restored.")
        print('Initialized')
        feed_dict = {
            tf_dataset: dataset,
        }
        predictions = sess.run(variables['layers'][3], feed_dict=feed_dict)
        print(predictions)
        print(LETTERS[np.argmax(predictions)])


def predict_pb(dataset):
    graph_def = graph_pb2.GraphDef()
    with open('./tmp/output_graph.pb', "rb") as f:
        graph_def.ParseFromString(f.read)

    for node in graph_def.node:
        print(node)

    tf_dataset = tf.placeholder(tf.float32, shape=(1, IMAGE_PIXELS))

    with tf.Session(graph=graph_def) as sess:
        # saver.restore(sess, './save/nn_2_layer.ckpt')

        # Init handler
        print("Model restored.")
        print('Initialized')
        feed_dict = {
            tf_dataset: dataset,
        }
        predictions = sess.run("", feed_dict=feed_dict)
        print(predictions)
        print(LETTERS[np.argmax(predictions)])

dataset = load_letter('./letters/letter.png')
print(dataset.shape)
flat_dataset = np.zeros(shape=(1, 784))
flat_dataset[0, :] = dataset.ravel()
print(flat_dataset.shape)
predict_pb(flat_dataset)

