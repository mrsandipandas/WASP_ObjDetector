# Object Detector

## Installation

1. Create the environment using environment.yaml
2. Install tensorflow models
```
git clone https://github.com/tensorflow/models
```
3. COCO API installation:
```
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools <path_to_tensorflow_models>/models/research/
```
4. Protobuf Compilation
  - Goto <path_to_tensorflow_models>/models/research/
```
protoc object_detection/protos/*.proto --python_out=.
```



