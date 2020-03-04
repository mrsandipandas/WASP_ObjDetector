import cv2
import numpy as np

class ObjDetector:
    def __init__(self, show_inp_video=True):
        self.model = []
        self.show_inp_video = show_inp_video

    # Do inferencing on the frames using the trained NN model
    def process_frame(self):
        pass

    # Read the camera data from webcam/file
    def read_frames(self):
        cap = cv2.VideoCapture(0)

        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Our operations on the frame come here
            #color = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if self.show_inp_video:
                # Display the resulting frame
                cv2.imshow('frame',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

# Simple example
o = ObjDetector(True)
o.read_frames()

