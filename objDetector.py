import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
import os

class ObjDetector:
    # Tensorflow patch for compatibility
    utils_ops.tf = tf.compat.v1
    tf.gfile = tf.io.gfile

    def __init__(self, file_dir='', file_name='', show_inp_video=True, img_row=480, img_col=640, channel=3):
        # The NN pre-trained model with transfer learning
        self.model = None
        self.category_index = None
        self.model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
        self.label_path = '/home/sandy/Projects/PhD/Courses/WASP/models/research/object_detection/data/mscoco_label_map.pbtxt'
        self.show_inp_video = show_inp_video
        self.file_name = file_name
        self.file_dir = os.path.join(os.getcwd(), file_dir)
        self.file_path = os.path.join(self.file_dir, self.file_name)
        self.img_row = img_row
        self.img_col = img_col
        self.channel = channel
    
    
    def load_model(self): 
        base_url = 'http://download.tensorflow.org/models/object_detection/'
        model_file = self.model_name + '.tar.gz'
        model_dir = tf.keras.utils.get_file(
            fname=self.model_name, 
            origin=base_url + model_file,
            untar=True)

        model_dir = os.path.join(model_dir, "saved_model")
        model = tf.saved_model.load(str(model_dir))
        model = model.signatures['serving_default']
        self.model = model
        self.category_index = label_map_util.create_category_index_from_labelmap(self.label_path, use_display_name=True)
    
    # Do inferencing on the frames using the trained NN model
    def process_frame(self, frame):
        # Work on the frame here by applying different model 
        # Dimension of the matrix 480, 640, 3
        img_shape = (self.img_row, self.img_col, self.channel)
        if self.model is None:
            # self.model = tf.keras.applications.MobileNetV2(input_shape=img_shape, include_top=False, weights='imagenet')
            self.load_model()

        # Converting image to numpy array   
        image = np.asarray(frame)
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis,...]
        
        # Run inference
        inference_dict = self.model(input_tensor)
        
        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(inference_dict.pop('num_detections'))
        inference_dict = {key:value[0, :num_detections].numpy() for key,value in inference_dict.items()}
        inference_dict['num_detections'] = num_detections

        # detection_classes should be ints.
        inference_dict['detection_classes'] = inference_dict['detection_classes'].astype(np.int64)
    
        # Handle models with masks:
        if 'detection_masks' in inference_dict:
            # Reframe the the bbox mask to the image size.
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                                        inference_dict['detection_masks'], inference_dict['detection_boxes'],
                                        image.shape[0], image.shape[1])      
            detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
            inference_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

        return inference_dict

    # Read the camera data from webcam/file
    def open_video_stream(self): 
        if os.path.isfile(self.file_path):
            cap = cv2.VideoCapture(self.file_path)
        
        else:
            cap = cv2.VideoCapture(0)

        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Our operations on the frame come here
            #color = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            inference_dict = self.process_frame(frame)

            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                inference_dict['detection_boxes'],
                inference_dict['detection_classes'],
                inference_dict['detection_scores'],
                self.category_index,
                instance_masks=inference_dict.get('detection_masks_reframed', None),
                use_normalized_coordinates=True,
                line_thickness=8
            )

            if self.show_inp_video:
                # Display the resulting frame
                # Uncomment if you want window rescaling
                # cv2.namedWindow("Real time classification", cv2.WINDOW_NORMAL)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'Press q to exit', (10,40), font, 0.75, (0, 0, 255), 1, cv2.LINE_AA)
                frame = cv2.resize(frame, (self.img_col, self.img_row))                
                cv2.imshow('Real time classification based on coco dataset',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

# Just a simple example
o = ObjDetector(show_inp_video=True, img_row=480, img_col=640)
o.open_video_stream()

