from scipy import ndimage

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
        saver.restore(sess, '../save/nn_2_layer.ckpt')

        # Init handler
        print("Model restored.")
        print('Initialized')
        feed_dict = {
            tf_dataset: dataset,
        }
        predictions = sess.run(variables['layers'][3], feed_dict=feed_dict)
        print(predictions)
        print(LETTERS[np.argmax(predictions)])
        [print(n.name) for n in tf.get_default_graph().as_graph_def().node]


def predict_pb(dataset):

    graph = retrieve_graph()
    [print(n.name) for n in graph.as_graph_def().node]
    tf_dataset = tf.placeholder(tf.float32, shape=(1, IMAGE_PIXELS))

    y = graph.get_tensor_by_name('prefix/output_node:0')
    x = graph.get_tensor_by_name('prefix/input_node:0')

    with tf.Session(graph=graph) as sess:
        # saver.restore(sess, './save/nn_2_layer.ckpt')

        # Init handler
        print("Model restored.")
        print('Initialized')
        feed_dict = {
            x: dataset,
        }
        predictions = sess.run(y, feed_dict=feed_dict)
        print(predictions)
        print(LETTERS[np.argmax(predictions)])


def retrieve_graph():
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile("../tmp/frozen_model.pb", "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph

dataset = load_letter('../letters/letter.png')
print(dataset.shape)
# flat_dataset = np.zeros(shape=(128, 784))
# flat_dataset[0, :] = dataset.ravel()
# print(flat_dataset.shape)
# predict_pb(flat_dataset)

predict_pb(dataset.ravel())
