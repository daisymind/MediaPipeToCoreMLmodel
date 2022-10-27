# toFrozen_model.py
#
import tensorflow as tf
 
FROZEN_MODEL_FILE = 'frozen_model.pb' # conversion target 'pb' name
 
# Taken from https://stackoverflow.com/a/52823701/4708657
def freeze_graph(graph, session, output):
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
        graph_io.write_graph(graphdef_frozen, ".", FROZEN_MODEL_FILE, as_text=False)
 
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="selfie_segmentation.tflite")
interpreter.allocate_tensors()
 
input_shape = [256, 256, 3] # set shape of input for tflite
model = network(input_shape, interpreter)
 
# FREEZE GRAPH
session = tf.keras.backend.get_session()
 
INPUT_NODE = model.inputs[0].op.name
OUTPUT_NODE = model.outputs[0].op.name
freeze_graph(session.graph, session, [out.op.name for out in model.outputs])
