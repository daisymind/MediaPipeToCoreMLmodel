# convert_inputType.py :convert multiArray to image type
import coremltools as ct
from coremltools.proto import FeatureTypes_pb2 as ft
 
spec = ct.utils.load_spec("model_coreml_float32.mlmodel") # miltiArray type
builder = ct.models.neural_network.NeuralNetworkBuilder(spec=spec)
 
# change the input so the model can accept 256x256 RGB images
input = spec.description.input[0]
input.type.imageType.width = 256
input.type.imageType.height = 256
input.type.imageType.colorSpace = ft.ImageFeatureType.RGB
 
# converted input/output features
builder.inspect_input_features()
builder.inspect_output_features()
 
# save inputType-converted model
ct.utils.save_spec(spec, 'selfie_segmentation.mlmodel')
