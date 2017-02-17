import tensorflow as tf
import numpy as np

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_LABELS = 10


def inference(tf_train_dataset):
    """
    Builds the graph as far as is required for running the network forward to make predictions
    :param tf_train_dataset:
    :return: Output tensor with the computed logits
    """
    # Variables.
    # These are the parameters that we are going to be training. The weight
    # matrix will be initialized using random values following a (truncated)
    # normal distribution. The biases get initialized to zero.
    weights = tf.Variable(tf.truncated_normal([IMAGE_PIXELS, NUM_LABELS]), name='weights')
    biases = tf.Variable(tf.zeros([NUM_LABELS]), name='biases')

    # Training computation.
    # We multiply the inputs with the weight matrix, and add biases. We compute
    # the softmax and cross-entropy (it's one operation in TensorFlow, because
    # it's very common, and it can be optimized).
    logits = tf.matmul(tf_train_dataset, weights) + biases

    return logits


def loss(logits, tf_train_labels):
    """
    Adds to the inference graph the ops required to generate loss
    :return: Loss tensor of type float
    """

    # We take the average of this cross-entropy across all training examples: that's our loss.
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)
    return tf.reduce_mean(cross_entropy)


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