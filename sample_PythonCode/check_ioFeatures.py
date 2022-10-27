# check_ioFeatures.py :check input/output features
import coremltools as ct
 
spec = ct.utils.load_spec("model_coreml_float32.mlmodel") # converted mlmodel
builder = ct.models.neural_network.NeuralNetworkBuilder(spec=spec)
 
builder.inspect_input_features()
builder.inspect_output_features()
