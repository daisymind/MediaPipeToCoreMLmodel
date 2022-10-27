## How to get Apple CoreML model converted from Google MediaPipe

Here you will find information on how to get the converted mlmodel from Google MediaPipe, and on using the converted mlmodel with 'SelfieSegmentation' as an example.


### Target: What we want to get
Apple CoreML 'mlmode' that converted from 'tflite' of MediaPipe model.

We want to use MediaPipe's small, fast, real-time processing-friendly model for iOS apps.

You can get 'tflite' model from [Google MediaPipe](https://google.github.io/mediapipe/).

### Quest for conversion methods: it's trickey :confused:
You will probably find several conversion methods from Net, but we cannot find a way to convert directly to mlmodel.

The major conversion methods found are considered as follows.
- a) First, convert tflite to protbuf,pb using Tensorflow, then convert pb to mlmodel using Apple coremltools.
- b) Using TensorFlowLitePod with [CocoaPods](https://cocoapods.org), this method is no mlmodel but activate tflite model on iOS.
- c) Using [TFCoreML](https://github.com/tf-coreml/tf-coreml), Unclear if tflite model can be converted.

For us, Method 'a)' seems like a good way to go.
And I found a conversion python code that looks good.

Ready for conversion:
- get tflite model that you wish to convert from Google MediaPipe
- using Python, Tensorflow environment
- convert from 'tflite' to 'pb' with Tensorflow, later convert to mlmodel with coremltools

Here is a sample python code below, reference from [this site](https://blog.xmartlabs.com/2019/11/22/TFlite-to-CoreML/), this convert tflite model file "selfie_segmentation.tflite" at [Google MediaPipe](https://google.github.io/mediapipe/) into pb file. Before conversion, check and set input_shape of tflite with a tool such as [Netron](https://github.com/lutzroeder/netron).

```python
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
```

Run python code to convert, Unfortunately, I encountered an error.

```bash
# terminal mode: shell command e.g.bash
% python toFrozen_model.py
   :
RuntimeError: Encountered unresolved custom op: Convolution2DTransposeBias.Node number 244 (Convolution2DTransposeBias) failed to prepare.
```

After examining this error, it appears that there is a special node, custom op like Convolution2DTransposeBias, in the tflite model. Furthermore, it seems that there are other custom ops.

This means that the custom op must be added for conversion. This is getting rather tricky.

### More quest and finally hit it! :thumbsup:
I came across a website that solves the problem. That site publishes a docker environment that solves the custom op issue and enables model conversion.

What is even more exciting is that they had already published the converted mlmodel from MediaPipe. Thanks! to the author.

#### How to get it
[GitHub tflite2tensorflow](https://github.com/PINTO0309/tflite2tensorflow) publishes a Docker environment that enables tflite to multi-model conversion. CoreML mlmodel is also supported.

Easy way to get Models that has already been converted:
- Check [GitHub PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo) where the converted models are available
- Search by model-name same as it appears in MediaPipe ,e.g. Selfie_Segmentation
- Click on Link for the desired model to go to the folder
- Get file "download.sh"

Run "download.sh" in your local environment.

```bash
# terminal mode: shell command e.g.bash
% chmod +x download.sh
% ./download.sh
```
Now, you can get the mlmodel converted from MediaPipe.

For example, in case of Selfie_Segmentation, the following files will be downloaded. You will find `model_coreml_float32.mlmodel` in there.:smile:
```
% ls
cookie
download.sh
saved_model_openvino/
saved_model_tflite_tfjs_tftrt_onnx_coreml/
selfiesegmentation_mlkit-256x256-2021_01_19-v1215.f16.json
selfiesegmentation_mlkit-256x256-2021_01_19-v1215.f16.tflite

% ls saved_model_tflite_tfjs_tftrt_onnx_coreml             
model_coreml_float32.mlmodel     model_full_integer_quant.tflite  tensorrt_saved_model_float16/
model_float16_quant.tflite       model_integer_quant.tflite       tensorrt_saved_model_float32/
model_float32.onnx               model_weight_quant.tflite        tfjs_model_float16/
model_float32.pb                 multiArrayToImage.ipynb          tfjs_model_float32/
model_float32.tflite             saved_model.pb                   variables/

```
### Something to be concerned about :worried:
In the case of the Selfie_Segmentation model, the input/output of the converted mlmodel is of type multiArray, and it is said that a simple conversion from Tensorflow would result in a multiArray.

Here is the sample code for checking input/output type of mlmodel with coremltools.
```python
# check_ioFeatures.py :check input/output features
import coremltools as ct

spec = ct.utils.load_spec("model_coreml_float32.mlmodel") # converted mlmodel
builder = ct.models.neural_network.NeuralNetworkBuilder(spec=spec)

builder.inspect_input_features()
builder.inspect_output_features()
```
```
[Id: 0] Name: input_1
          Type: multiArrayType {
  shape: 1
  shape: 256
  shape: 256
  shape: 3
  dataType: FLOAT32
}

[Id: 0] Name: activation_10
          Type: multiArrayType {
  dataType: FLOAT32
}
```
When using this multiArray type model in Swift with CoreML, we need to consider the conversion between image type and multiArray type.

Reference:
- [Image Input and Output :Apple](https://coremltools.readme.io/docs/image-inputs)
- [How to convert images to MLMultiArray](https://machinethink.net/blog/coreml-image-mlmultiarray/)
- [Support for image outputs :Apple developerForum](https://developer.apple.com/forums/thread/81571)
- [MLMultiArray to image conversion :CoreMLHelpers at GitHub](https://github.com/hollance/CoreMLHelpers/blob/master/Docs/MultiArray2Image.markdown)

### Appendix: quest for problem-solving :dizzy_face:
Checking the input/output type of the converted 'Selfie_Segmentation' mlmodel, it is MultiArray. Since it is difficult to use it as it is, I tried to convert MultiArray to image type using coremltools.

But,something wrong :cry:.  When I load the converted image-type mlmodel into project in xcode, CoreML throws a compilation error. On the other hand, the multiArray-type model doesn't throw any error.

I'm currently working on a solution, but it is still open.

Here is the input-type conversion code with coremltools. However, something may be missing.
```python
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
```
```
[Id: 0] Name: input_1
          Type: imageType {
  width: 256
  height: 256
  colorSpace: RGB
}

[Id: 0] Name: activation_10
          Type: multiArrayType {
  dataType: FLOAT32
}
```
Input type is converted from multiArray to image, and new mlmodel 'selfie_segmentation.mlmodel' is saved.

Unfortunately, this 'selfie_segmentation.mlmodel' causes CoreML to throw the following error.
```
Espresso exception: "Invalid blob shape": generic_elementwise_kernel: cannot broadcast:

----------------------------------------

SchemeBuildError: Failed to build the scheme "testSelfieSegmentation"

compiler error:  Espresso exception: "Invalid blob shape": generic_elementwise_kernel: cannot broadcast:

Compile CoreML model selfie_segmentation.mlmodel:
coremlc: error: compiler error:  Espresso exception: "Invalid blob shape": generic_elementwise_kernel: cannot broadcast:
 (1, 16, 8, 128)
 (1, 16, 2, 128)
```

I don't know how to resolve this yet.

---
Can anyone please give me some tips on how to solve this CoreML compile-error problem?
