import numpy as np
import tensorflow as tf

# Reformat into a TensorFlow-friendly shape:
# convolutions need the image data formatted as a cube (width by height by #channels)
# labels as float 1-hot encodings.

# TODO These values from shared config
image_size = 28
num_labels = 10
num_channels = 1  # grayscale
patch_size = 5
depth = 16
num_hidden = 64


# TODO Refactor. This function "private"? or somewhere else
def reformat(dataset, labels):
    dataset = dataset.reshape(
        (-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


def inference():
    """
    Build the variables to be used by the model
    :return: The variables in the model
    """
    weights_1 = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1), name="W1")
    biases_1 = tf.Variable(tf.zeros([depth]), name="B1")

    weights_2 = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1), "W2")
    biases_2 = tf.Variable(tf.constant(1.0, shape=[depth]), name="B2")

    weights_3 = tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1),
                            name="W3")
    biases_3 = tf.Variable(tf.constant(1.0, shape=[num_hidden]), name="B3")

    weights_4 = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1), "W4")
    biases_4 = tf.Variable(tf.constant(1.0, shape=[num_labels]), name="B4")

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

    inference_components = {
        'weights': weights,
        'biases': biases,
    }

    return inference_components


def model(data, inference_components):
    weights = inference_components['weights']
    biases = inference_components['biases']

    weights_1 = weights['hidden_1']
    weights_2 = weights['hidden_2']
    weights_3 = weights['hidden_3']
    weights_4 = weights['out']
    biases_1 = biases['hidden_1']
    biases_2 = biases['hidden_2']
    biases_3 = biases['hidden_3']
    biases_4 = biases['out']

    stride = [1, 2, 2, 1] # tf.nn.max_pool(1, 2, 2, )
    conv = tf.nn.conv2d(data, weights_1, stride, padding='SAME')
    hidden = tf.nn.relu(conv + biases_1)
    conv = tf.nn.conv2d(hidden, weights_2, stride, padding='SAME')
    hidden = tf.nn.relu(conv + biases_2)
    shape = hidden.get_shape().as_list()
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, weights_3) + biases_3)
    return tf.matmul(hidden, weights_4) + biases_4


def loss(train_labels, logits):
    loss_value = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=train_labels, logits=logits))
    return loss_value


def training(learning_rate, loss_value):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_value)
    return optimizer


def prediction(tf_train_dataset, tf_valid_dataset, tf_test_dataset, inference_components):

    valid_logits = model(tf_valid_dataset, inference_components)
    test_logits = model(tf_test_dataset, inference_components)
    logits = model(tf_train_dataset, inference_components)

    train_prediction = tf.nn.softmax(logits, name='output_node')
    valid_prediction = tf.nn.softmax(valid_logits)
    test_prediction = tf.nn.softmax(test_logits)

    return train_prediction, valid_prediction, test_prediction


def evaluation(predictions, labels):
    """
    Evaluate the quality of the logits at predicting the label.
    Args:
        predictions: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size], with values in the
          range [0, NUM_CLASSES).
    Returns:
        A scalar int32 representing the percentage of correct predictions
    """
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]