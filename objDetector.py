from __future__ import absolute_import, division, print_function, unicode_literals
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

import tensorflow as tf

import tensorflow_datasets as tfds
tfds.disable_progress_bar()



class ObjDetector:

    def __init__(self, file_dir='', file_name='', show_inp_video=True):
        self.model = []
        self.show_inp_video = show_inp_video
        self.file_name = file_name
        self.file_dir = os.path.join(os.getcwd(), file_dir)
        self.file_path = os.path.join(self.file_dir, self.file_name)
    
    # Do inferencing on the frames using the trained NN model
    def process_frame(self, frame):
        # Work on the frame here by applying different model
        print (frame)

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

            self.process_frame(frame)

            if self.show_inp_video:
                # Display the resulting frame
                cv2.imshow('frame',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

    def format_example(self, image, label):
        IMG_SIZE = 160
        image = tf.cast(image, tf.float32)
        image = (image/127.5) - 1
        image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
        return image, label
    
    def getDatasets(self):
        (self.raw_train, self.raw_validation, self.raw_test), self.metadata = tfds.load(
            'cats_vs_dogs',
            split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
            with_info=True,
            as_supervised=True,
            )
        self.get_label_name = self.metadata.features['label'].int2str

        # for image, label in self.raw_train.take(2):
        #     plt.figure()
        #     plt.imshow(image)
        #     plt.title(self.get_label_name(label))
        #     plt.show()
        
        self.train = self.raw_train.map(self.format_example)
        self.validation = self.raw_validation.map(self.format_example)
        self.test = self.raw_test.map(self.format_example)

        BATCH_SIZE = 32
        SHUFFLE_BUFFER_SIZE = 1000

        self.train_batches = self.train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
        self.validation_batches = self.validation.batch(BATCH_SIZE)
        self.test_batches = self.test.batch(BATCH_SIZE)

        # Create the base model from the pre-trained model MobileNet V2
        IMG_SIZE = 160
        IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
        self.base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                    include_top=False,
                                                    weights='imagenet')

        self.base_model.trainable = False
        self.base_model.summary()
        self.global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        self.prediction_layer = tf.keras.layers.Dense(1)
        
        for image_batch, label_batch in self.train_batches.take(1):
            feature_batch = self.base_model(image_batch)
            print(feature_batch.shape)
            feature_batch_average = self.global_average_layer(feature_batch)
            print(feature_batch_average.shape)
            prediction_batch = self.prediction_layer(feature_batch_average)
            print(prediction_batch.shape)
        
        self.model = tf.keras.Sequential([self.base_model,
                                          self.global_average_layer,
                                          self.prediction_layer
                                          ])

        base_learning_rate = 0.0001
        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
        
        self.model.summary()

        initial_epochs = 10
        validation_steps=20

        loss0,accuracy0 = self.model.evaluate(self.validation_batches, steps = validation_steps)

        print("initial loss: {:.2f}".format(loss0))
        print("initial accuracy: {:.2f}".format(accuracy0))

        history = self.model.fit(self.train_batches,
                    epochs=initial_epochs,
                    validation_data=self.validation_batches)


        ## Plotting 
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()),1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0,1.0])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.show()

        ## Go again
        self.base_model.trainable = True
        # Let's take a look to see how many layers are in the base model
        print("Number of layers in the base model: ", len(self.base_model.layers))

        # Fine-tune from this layer onwards
        fine_tune_at = 100

        # Freeze all the layers before the `fine_tune_at` layer
        for layer in self.base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        self.model = tf.keras.Sequential([self.base_model,
                                          self.global_average_layer,
                                          self.prediction_layer
                                          ])

        self.model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])

        self.model.summary()

        print("Trainable variables #: ", len(self.model.trainable_variables))

        fine_tune_epochs = 10
        total_epochs =  initial_epochs + fine_tune_epochs

        history_fine = self.model.fit(self.train_batches,
                         epochs=total_epochs,
                         initial_epoch =  history.epoch[-1],
                         validation_data=self.validation_batches)

        acc += history_fine.history['accuracy']
        val_acc += history_fine.history['val_accuracy']

        loss += history_fine.history['loss']
        val_loss += history_fine.history['val_loss']

        ## Plotting after re-train
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.ylim([0.8, 1])
        plt.plot([initial_epochs-1,initial_epochs-1],
                plt.ylim(), label='Start Fine Tuning')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.ylim([0, 1.0])
        plt.plot([initial_epochs-1,initial_epochs-1],
                plt.ylim(), label='Start Fine Tuning')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.show()


        

        

        


    
            

# Just a simple example
o = ObjDetector(show_inp_video=True)
o.getDatasets() 

