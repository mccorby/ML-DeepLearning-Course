from convnet import *

# TODO These values from shared config
image_size = 28
num_labels = 10
num_channels = 1  # grayscale

# Let's build a small network with two convolutional layers, followed by one fully connected layer.'
# Convolutional networks are more expensive computationally, so we'll limit its depth and number
# of fully connected nodes.

# TODO Constants. These values from config or shared config
batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64


def run_training(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels):
    graph = tf.Graph()

    with graph.as_default():
        # Input data.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels),
                                          name='input_node')
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # Variables.
        variables = inference()

        # Training computation.
        logits = model(tf_train_dataset, variables)

        loss_value = loss(tf_train_labels, logits)

        # Optimizer.
        optimizer = training(0.05, loss_value)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(model(tf_valid_dataset, variables))
        test_prediction = tf.nn.softmax(model(tf_test_dataset, variables))

    num_steps = 1001

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print('Initialized')
        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, l, predictions = session.run([optimizer, loss_value, train_prediction], feed_dict=feed_dict)
            if step % 50 == 0:
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % evaluation(predictions, batch_labels))
                print('Validation accuracy: %.1f%%' % evaluation(valid_prediction.eval(), valid_labels))
        print('Test accuracy: %.1f%%' % evaluation(test_prediction.eval(), test_labels))
