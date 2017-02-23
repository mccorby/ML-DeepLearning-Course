import numpy as np
from scipy import ndimage
import tensorflow as tf
from nn_2_layers import *

IMAGE_SIZE = 28
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


# def predict(image_file):
#     sess = tf.Session()
#     new_saver = tf.train.import_meta_graph('./save/nn_2_layer')
#     new_saver.restore(sess, tf.train.latest_checkpoint('./save/checkpoint'))
#     all_vars = tf.get_collection('vars')
#     for v in all_vars:
#         v_ = sess.run(v)
#         print(v_)

BATCH_SIZE = 128

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_LABELS = 10
LAYER1_NODES = 1024
LAYER2_NODES = 256
LAYER3_NODES = 128


def predict():

    # Build the graph
    graph = tf.Graph()
    with graph.as_default():

        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(tf.float32,
                                          shape=(BATCH_SIZE, IMAGE_PIXELS))
        # tf_train_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))
        # tf_valid_dataset = tf.constant(valid_dataset)
        # tf_test_dataset = tf.constant(test_dataset)
        # global_step = tf.Variable(0)

        # Variables.
        variables = inference(tf_train_dataset)
        # Training computation.
        # loss_op = loss(tf_train_labels, variables, 0.05)

        # Optimizer.
        # learning_rate = tf.train.exponential_decay(training_rate, global_step, 1000, 0.65, staircase=True)
        # train_op = training(loss_op, learning_rate, global_step)

        # Predictions for the training, validation, and test data.
        # train_prediction, valid_prediction, test_prediction = prediction(tf_valid_dataset,
        #                                                                  tf_test_dataset, variables)

        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        saver.restore(sess, './save/nn_2_layer.ckpt')
        # Init handler
        print("Model restored.")
        print('Initialized')
        w1 = sess.run(variables['weights']['hidden_1'])
        print(w1)


predict()
