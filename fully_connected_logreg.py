import tensorflow as tf
import load_data
import logistic_regression


def run_training():
    # Get the sets of images and labels for training, validation, and
    # test on MNIST.

    # TODO Get this value from config
    data_root = '/Users/jco59/ML/WorkingData/DL-Course/assignment1'

    data_sets = load_data.load_data(data_root)

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        logits = logistic_regression.inference(data_sets['train_dataset'])
