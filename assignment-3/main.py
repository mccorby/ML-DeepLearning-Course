import load_data
import numpy as np
import tensorflow as tf

data_root = '/Users/jco59/ML/WorkingData/DL-Course/assignment1'
train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = load_data.load_data(data_root)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

'''
Reformat into a shape that's more adapted to the models we're going to train:
data as a flat matrix,
labels as float 1-hot encodings.
'''
image_size = 28
num_labels = 10

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


'''
Training a multinomial logistic regression with TensorFlow
'''

'''
First you describe the computation that you want to see performed:
what the inputs, the variables, and the operations look like.
These get created as nodes over a computation graph.

Then you can run the operations on this graph as many times as you want
by calling session.run(), providing it outputs to fetch from the graph that get returned.
'''

reg_param = 0.001

# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.
train_subset = 10000

graph = tf.Graph()
with graph.as_default():
    # Input data.
    # Load the training, validation and test data into constants that are
    # attached to the graph.
    tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
    tf_train_labels = tf.constant(train_labels[:train_subset])
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    # These are the parameters that we are going to be training. The weight
    # matrix will be initialized using random values following a (truncated)
    # normal distribution. The biases get initialized to zero.
    weights = tf.Variable(
        tf.truncated_normal([image_size * image_size, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    # We multiply the inputs with the weight matrix, and add biases. We compute
    # the softmax and cross-entropy (it's one operation in TensorFlow, because
    # it's very common, and it can be optimized). We take the average of this
    # cross-entropy across all training examples: that's our loss.
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    loss += reg_param  * tf.nn.l2_loss(weights)

    # Optimizer.
    # We are going to find the minimum of this loss using gradient descent.
    optimizier = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    # These are not part of training, but merely here so that we can report
    # accuracy figures as we train.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)


'''
Let's run this computation and iterate!
'''
num_steps = 801

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

with tf.Session(graph=graph) as session:
    # This is a one-time operation which ensures the parameters get initialized as
    # we described in the graph: random weights for the matrix, zeros for the biases.
    tf.global_variables_initializer().run()
    print('Initialized!')
    for step in range(num_steps):
        # Run the computations. We tell .run() that we want to run the optimizer,
        # and get the loss value and the training predictions returned as numpy arrays.
        _, l, predictions = session.run([optimizier, loss, train_prediction])
        if (step % 100 == 0):
            print('{} {}: {}'.format('Loss at step', step, l))
            print('{}: {}'.format('Training Accuracy', accuracy(predictions, train_labels[:train_subset, :])))

            # Calling .eval() on valid_prediction is basically like calling run(), but
            # just to get that one numpy array. Note that it recomputes all its graph
            # dependencies.
            print('{}: {}'.format('Validation Accuracy', accuracy(valid_prediction.eval(), valid_labels)))

    print('{}: {}'.format('Test Accuracy', accuracy(test_prediction.eval(), test_labels)))


'''
Use now SGD
'''
# The graph will be similar, except that instead of holding all the training data into a constant node,
# we create a Placeholder node which will be fed actual data at every call of session.run().

batch_size = 128

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables
    weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))

    # Training computation
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    loss += reg_param  * tf.nn.l2_loss(weights)

    # Optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation and test data
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(test_dataset, weights) + biases)

# Let's run it
num_steps = 3001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized!')

    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a mini batch
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

        if (step % 500 == 0):
            print('{} {}: {}'.format('Minibatch loss at step', step, l))
            print('{}: {}'.format('Minibatch Accuracy', accuracy(predictions, batch_labels)))
            print('{}: {}'.format('Validation Accuracy', accuracy(valid_prediction.eval(), valid_labels)))

    print('{}: {}'.format('Test Accuray', accuracy(test_prediction.eval(), test_labels)))


'''
Use relu and a hidden layer
'''
H = 1024

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables
    weights_1 = tf.Variable(tf.truncated_normal([image_size * image_size, H], stddev=0.1))
    weights_2 = tf.Variable(tf.truncated_normal([H, num_labels], stddev=0.1))
    biases_1 = tf.Variable(tf.constant(0.1, shape=[H]))
    biases_2 = tf.Variable(tf.constant(0.1, shape=[num_labels]))

    hidden = tf.nn.relu(tf.matmul(tf_train_dataset, weights_1) + biases_1)
    logits = tf.matmul(hidden, weights_2) + biases_2

    # Training computation
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    # Adding regularization from both layers: output and hidden
    loss += reg_param * (tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2))

    # Optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation and test data
    train_prediction = tf.nn.softmax(logits)
    valid_logits = tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights_1) + biases_1), weights_2) + biases_2
    test_logits = tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights_1) + biases_1), weights_2) + biases_2
    valid_prediction = tf.nn.softmax(valid_logits)
    test_prediction = tf.nn.softmax(test_logits)

# Let's run it
num_steps = 3001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized!')

    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a mini batch
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

        if (step % 500 == 0):
            print('{} {}: {}'.format('Relu loss at step', step, l))
            print('{}: {}'.format('Relu Accuracy', accuracy(predictions, batch_labels)))
            print('{}: {}'.format('Relu Accuracy', accuracy(valid_prediction.eval(), valid_labels)))

    print('{}: {}'.format('Test Accuray', accuracy(test_prediction.eval(), test_labels)))
