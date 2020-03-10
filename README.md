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



