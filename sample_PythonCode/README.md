### Sample_PythonCode
##### run environment:
- Linux / MacOS terminal / GoogleColab
- Python, Tensorflow, coremltools

##### Sample code:
- **toFrozen_model.py**
 - convert tensorflow model to protbuf,pb
- **check_ioFeatures.py**
 - check type of input/output for mlmodel
- **convert_inputType.py**
 - convert input type of multiArray to image with coremltools

##### Model files:
- **selfie_segmentation.tflite**
 - TFlite model from [Google MediaPipe](https://google.github.io/mediapipe/)
- **model_coreml_float32.mlmodel**
 - mlmodel file from [GitHub PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo)
- **selfie_segmentation.mlmodel**
 - mlmodel file converted by convert_inputType.py
