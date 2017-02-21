import tensorflow as tf
import numpy as np

# TODO Some of the functions in this module are the same as in other (logistic_regression).
# inference(), prediction() differ


IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_LABELS = 10
LAYER_SIZE = 1024


def inference(tf_train_dataset):
    """
    Builds the graph as far as is required for running the network forward to make predictions
    :param tf_train_dataset:
    :return: Output tensor with the computed logits, weights and biases used in the different layers
    """
    weights_1 = tf.Variable(tf.truncated_normal([IMAGE_PIXELS, LAYER_SIZE], stddev=0.1))
    weights_2 = tf.Variable(tf.truncated_normal([LAYER_SIZE, NUM_LABELS], stddev=0.1))
    biases_1 = tf.Variable(tf.constant(0.1, shape=[LAYER_SIZE]))
    biases_2 = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

    weights = {
        'hidden_1': weights_1,
        'out': weights_2
    }

    biases = {
        'hidden_1': biases_1,
        'out': biases_2
    }

    hidden = tf.nn.relu(tf.matmul(tf_train_dataset, weights_1) + biases_1)
    logits = tf.matmul(hidden, weights_2) + biases_2

    return logits, weights, biases


def loss(logits, tf_train_labels, reg_beta, weights):
    """
    Adds to the inference graph the ops required to generate loss
    :return: Loss tensor of type float
    """

    # We take the average of this cross-entropy across all training examples: that's our loss.
    reg_term = reg_beta * ((tf.nn.l2_loss(weights['hidden_1'])) + (tf.nn.l2_loss(weights['out'])))
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)

    return tf.reduce_mean(cross_entropy) + reg_term


def training(loss, learning_rate):
    """
    Adds to the loss graph the ops required to compute and apply gradients
    Creates a summarizer to track the loss over time in TensorBoard.
    Creates an optimizer and applies the gradients to all trainable variables.
    :return: The Op for training
    """
    # Optimizer.
    # We are going to find the minimum of this loss using gradient descent.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    train_op = optimizer.minimize(loss)
    return train_op


def evaluation(logits, labels):
    """
    Evaluate the quality of the logits at predicting the label.
    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size], with values in the
          range [0, NUM_CLASSES).
    Returns:
        A scalar int32 representing the percentage of correct predictions
    """
    return 100.0 * np.sum(np.argmax(logits, 1) == np.argmax(labels, 1)) / logits.shape[0]


def prediction(logits, tf_valid_dataset, tf_test_dataset, weights, biases):
    """
    # Predictions for the training, validation and test data
    :param logits:
    :param tf_valid_dataset:
    :param tf_test_dataset:
    :param weights:
    :param biases:
    :return:
    """
    weights_1 = weights['hidden_1']
    weights_2 = weights['out']
    biases_1 = biases['hidden_1']
    biases_2 = biases['out']
    train_prediction = tf.nn.softmax(logits)
    valid_logits = tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights_1) + biases_1), weights_2) + biases_2
    test_logits = tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights_1) + biases_1), weights_2) + biases_2
    valid_prediction = tf.nn.softmax(valid_logits)
    test_prediction = tf.nn.softmax(test_logits)

    return train_prediction, valid_prediction, test_prediction
