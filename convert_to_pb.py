import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import os

def convert_to_protobuf(sess, saver):
    input_graph_name = "inputGraph"
    input_saver_def_path = ""
    input_binary = False
    checkpoint_path = './save/nn_2_layer.ckpt'
    output_node_names = 'output_node'
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_graph_name = "output_graph.pb"
    output_graph_path = os.path.join("./tmp", output_graph_name)
    clear_devices = False

    tf.train.write_graph(sess.graph, "./tmp", input_graph_name)
    input_graph_path = os.path.join("./tmp", input_graph_name)
    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, checkpoint_path, output_node_names,
                              restore_op_name, filename_tensor_name,
                              output_graph_path, clear_devices, "")

output_graph = "./tmp/frozen_model.pb"

# We retrieve our checkpoint fullpath
checkpoint = tf.train.get_checkpoint_state('./save')
input_checkpoint = checkpoint.model_checkpoint_path
saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()


with tf.Session() as sess:
    saver.restore(sess, './save/nn_2_layer.ckpt')
    [print(n.name) for n in graph.as_graph_def().node]
    # We use a built-in TF helper to export variables to constants
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, # The session is used to retrieve the weights
        input_graph_def, # The graph_def is used to retrieve the nodes
        'output_node'.split(",") # The output node names are used to select the usefull nodes
    )

    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))

