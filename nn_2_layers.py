import tensorflow as tf
import numpy as np

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_LABELS = 10
LAYER1_NODES = 1024
LAYER2_NODES = 256
LAYER3_NODES = 128


def inference(tf_train_dataset):
    """
    We're not using dropout in this version to simplify the parameters.
    TODO Future versions will see the ability of adding whenever you want
    :param tf_train_dataset:
    :return:
    """
    # Define the weights and biases for the different layers
    # We use truncate normal distribution this is the recommended initializer for neural network weights and filters.

    # NOTE: If name is not specified it does not restore the model properly
    # (there are other ways but this is the easiest)
    weights_1 = tf.Variable(tf.truncated_normal([IMAGE_PIXELS, LAYER1_NODES], stddev=1.0 / np.sqrt(IMAGE_PIXELS)),
                            name="W1")
    biases_1 = tf.Variable(tf.zeros([LAYER1_NODES]), name="B1")

    weights_2 = tf.Variable(tf.truncated_normal([LAYER1_NODES, LAYER2_NODES], stddev=1.0 / np.sqrt(LAYER1_NODES)),
                            name="W2")
    biases_2 = tf.Variable(tf.zeros([LAYER2_NODES]), name="B2")

    weights_3 = tf.Variable(tf.truncated_normal([LAYER2_NODES, LAYER3_NODES], stddev=1.0 / np.sqrt(LAYER2_NODES)),
                            name="W3")
    biases_3 = tf.Variable(tf.zeros([LAYER3_NODES]), name="B3")

    weights_4 = tf.Variable(tf.truncated_normal([LAYER3_NODES, NUM_LABELS], stddev=1.0 / np.sqrt(LAYER3_NODES)),
                            name="W4")
    biases_4 = tf.Variable(tf.zeros([NUM_LABELS]), name="B4")

    weights = {
        'hidden_1': weights_1,
        'hidden_2': weights_2,
        'hidden_3': weights_3,
        'out': weights_4
    }

    biases = {
        'hidden_1': biases_1,
        'hidden_2': biases_2,
        'hidden_3': biases_3,
        'out': biases_4
    }

    # Training computations
    layer_1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights_1) + biases_1)
    layer_2 = tf.nn.relu(tf.matmul(layer_1, weights_2) + biases_2)
    layer_3 = tf.nn.relu(tf.matmul(layer_2, weights_3) + biases_3)
    logits = tf.matmul(layer_3, weights_4) + biases_4

    inference_components = {
        'weights': weights,
        'biases': biases,
        'layers': [layer_1, layer_2, layer_3, logits]
    }

    return inference_components


def loss(tf_train_labels, variables, reg_beta):
    # TODO This can be iterative
    weights1 = variables['weights']['hidden_1']
    weights2 = variables['weights']['hidden_2']
    weights3 = variables['weights']['hidden_2']
    weights4 = variables['weights']['out']
    logits = variables['layers'][3]

    reg_term = reg_beta * (tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2) + tf.nn.l2_loss(weights3)) + tf.nn.l2_loss(weights4)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)
    return tf.reduce_mean(cross_entropy) + reg_term


def training(loss, exponential_decay, global_step=None):
    """
    The learning rate is a parameter that determines how much an updating step influences
    the current value of the weights.
    While weight decay is an additional term in the weight update rule that causes
    the weights to exponentially decay to zero, if no other update is scheduled.

    :return:
    """
    return tf.train.GradientDescentOptimizer(exponential_decay).minimize(loss, global_step=global_step)


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


def prediction(tf_valid_dataset, tf_test_dataset, inference_components):
    """
    # Predictions for the training, validation and test data
    :param tf_valid_dataset:
    :param tf_test_dataset:
    :param inference_components:
    :return:
    """

    weights = inference_components['weights']
    biases = inference_components['biases']
    logits = inference_components['layers'][3]

    weights_1 = weights['hidden_1']
    weights_2 = weights['hidden_2']
    weights_3 = weights['hidden_3']
    weights_4 = weights['out']
    biases_1 = biases['hidden_1']
    biases_2 = biases['hidden_2']
    biases_3 = biases['hidden_3']
    biases_4 = biases['out']

    valid_l1 = tf.nn.relu(tf.matmul(tf_valid_dataset, weights_1) + biases_1)
    valid_l2 = tf.nn.relu(tf.matmul(valid_l1, weights_2) + biases_2)
    valid_l3 = tf.nn.relu(tf.matmul(valid_l2, weights_3) + biases_3)
    valid_logits = tf.matmul(valid_l3, weights_4) + biases_4

    test_l1 = tf.nn.relu(tf.matmul(tf_test_dataset, weights_1) + biases_1)
    test_l2 = tf.nn.relu(tf.matmul(test_l1, weights_2) + biases_2)
    test_l3 = tf.nn.relu(tf.matmul(test_l2, weights_3) + biases_3)
    test_logits = tf.matmul(test_l3, weights_4) + biases_4

    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(valid_logits)
    test_prediction = tf.nn.softmax(test_logits)

    return train_prediction, valid_prediction, test_prediction
