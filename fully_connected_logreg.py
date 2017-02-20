import tensorflow as tf
from logistic_regression import *


BATCH_SIZE = 128
MAX_STEPS = 3001


def run_training(training_rate, train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels):
    # Get the sets of images and labels for training, validation, and
    # test on MNIST.

    # Tell TensorFlow that the model will be built into the default Graph.
    graph = tf.Graph()
    with graph.as_default():

        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_PIXELS))
        tf_train_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))
        tf_valid_dataset = tf.constant(valid_dataset, name='valid_dataset')
        tf_test_dataset = tf.constant(test_dataset, name='test_dataset')

        # Build a Graph that computes predictions from the inference model.
        logits, weights, biases = inference(tf_train_dataset)

        # Add the ops for loss calculation to the graph
        loss_op = loss(logits, tf_train_labels)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = training(loss_op, training_rate)

        train_prediction, valid_prediction, test_prediction = prediction(logits, tf_valid_dataset,
                                                                         tf_test_dataset, weights, biases)

        # Add the Op to compare the logits to the labels during evaluation.
        # eval_correct = logistic_regression.evaluation(logits, tf_train_labels)

    with tf.Session(graph=graph) as sess:
        # Init handler
        init = tf.global_variables_initializer()

        # Run the Op to initialize the variables.
        sess.run(init)

        # Start the training loop.
        for step in range(MAX_STEPS):

            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)
            # Generate a mini batch
            batch_data = train_dataset[offset:(offset + BATCH_SIZE), :]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {
                tf_train_dataset: batch_data,
                tf_train_labels: batch_labels
            }
            _, loss_value, predictions = sess.run([train_op, loss_op, train_prediction], feed_dict=feed_dict)

            if step % 500 == 0:
                print('{} {}: {}'.format('Minibatch loss at step', step, loss_value))
                print('{}: {}'.format('Minibatch Accuracy', evaluation(predictions, batch_labels)))
                print('{}: {}'.format('Minibatch Validation Accuracy', evaluation(valid_prediction.eval(), valid_labels)))

        print('{}: {}'.format('Minibatch Test Accuracy', evaluation(test_prediction.eval(), test_labels)))


