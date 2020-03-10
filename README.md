# Object Detector

## Installation

1. Create the environment using environment.yaml

2. Clone to seperate modules
  - Tensorflow models
  - COCO API

3. Install tensorflow models
```
git clone https://github.com/tensorflow/models
```
4. COCO API installation:
```
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools <path_to_tensorflow_models>/models/research/
```
5. Protobuf Compilation
  - Goto <path_to_tensorflow_models>/models/research/
```
protoc object_detection/protos/*.proto --python_out=.
```
6. Add the paths to environment variables
```
export PYTHONPATH=$PYTHONPATH:<PATH_TO_TF>/TensorFlow/models/research
export PYTHONPATH=$PYTHONPATH:<PATH_TO_TF>/TensorFlow/models/research/object_detection
export PYTHONPATH=$PYTHONPATH:<PATH_TO_TF>/TensorFlow/models/research/slim
```

7. Running
```
python objDetector.py
```



