import numpy as np  
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import csv
import datetime
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #to suppress some warnings

if not os.path.exists('./client/data/emotion_data.csv'): # if file does not exist write header
    with open('./client/data/emotion_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "emotion", "pixels", "time stamp"])
      
# Get the full path of the current Python file
current_file = os.path.abspath(__file__)       

# check if the training data is present
if not os.path.join(os.path.dirname(current_file), 'data', 'train') or not os.path.join(os.path.dirname(current_file), 'data', 'test'):
    print("Please download the training data")
    sys.exit()
        
# Construct the path to the train directory
train_directory = os.path.join(os.path.dirname(current_file), 'data', 'train')
val_directory = os.path.join(os.path.dirname(current_file), 'data', 'test')

num_train = 28709 # number of training set
num_val = 7178 # number of validation set
batch_size = 64 #used for accuarcy

trainDataGenrator = ImageDataGenerator(rescale=1./255) # rescale the image between 0-1
valDataGenrator = ImageDataGenerator(rescale=1./255) # rescale the image between 0-1

train_generator = trainDataGenrator.flow_from_directory(
        train_directory,
        target_size=(48,48), 
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical'
    ) # Define the CNN Model Convolutional Neural Network 

validation_generator = valDataGenrator.flow_from_directory(
        val_directory,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale", 
        class_mode='categorical'
    ) # Define the CNN Model

# Create the model
model = Sequential() # Plot the training and validation loss + accuracy Initlizing the model
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1))) #
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# If you want to train the same model or try other models, go for this
def Train_Model(num_epoch):
    optimizer = Adam(learning_rate=0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model_info = model.fit(
            train_generator,
            steps_per_epoch=num_train // batch_size,
            epochs=num_epoch,
            validation_data=validation_generator,
            validation_steps=num_val // batch_size) #builds the model
    print(model_info)
    model.save_weights('./modules/model.h5')
    return True

class VideoCamera(object):
    isOpened = True  
    def __init__(self):
        self.video = cv2.VideoCapture(0, cv2.CAP_DSHOW) # 0 -> index of camera
    def __del__(self):
        self.video.release() # release the camera
    def get_frame(self):
        isOpened = True # check if camera is open
        model_file = os.path.join(os.path.dirname(current_file), 'modules', 'model.h5') # load the model file
        model.load_weights(model_file) # load the weights
        cv2.ocl.setUseOpenCL(False) # to avoid error
        emotion_dict = {0: "Angry", 1: "Disgusted", 
                2: "Fearful", 3: "Happy",
                4: "Neutral", 5: "Sad",
                6: "Surprised"} # dictionary of emotions
        ret, frame = self.video.read() # read the camera
        if not ret: # if not return the frame
            print("Video Capture error or Video feed ended")
            return None
        facecasc_file = os.path.join(os.path.dirname(current_file), 'modules', 'haarcascade_frontalface_default.xml') # load the cascade file
        facecasc = cv2.CascadeClassifier(facecasc_file) # load the cascade which is used to detect the face
        if frame is not None and len(frame) > 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to grayscale
            faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5) # detect the faces and store the positions 
            for (x, y, w, h) in faces: # frame, x, y, w, h
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 255, 0), 2) # draw rectangle to main frame 
                roi_gray = gray[y:y + h, x:x + w] # crop the region of interest i.e. face from the frame
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0) # resize the image
                prediction = model.predict(cropped_img) # predict the emotion
                maxindex = int(np.argmax(prediction)) # get the index of the largest value
                cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 2.55, 255), 2, cv2.LINE_AA) # write the emotion text above rectangle
                with open('./client/data/emotion_data.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([frame, emotion_dict[maxindex], roi_gray, datetime.datetime.now()])
            ret, jpeg = cv2.imencode('.jpg', frame) # encode the frame into jpeg
            if isOpened:
                return jpeg.tobytes() #byte array 64
        else:
            return None # return None if frame is empty

    def close_camera(self):
        isOpened = False
        self.video.release() # release the camera