import os
import sys
import cv2
import time
import torch
import argparse
import numpy as np 
import csv
import datetime

import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .Detection.Utils import ResizePadding
from .CameraLoader import CamLoader, CamLoader_Q
from .DetectorLoader import TinyYOLOv3_onecls

from .PoseEstimateLoader import SPPE_FastPose
from .fn import draw_single

from .Track.Tracker import Detection, Tracker
from .ActionsEstLoader import TSSTG

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #to suppress some warnings

      
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

def preproc(image, resize_fn):
    """
    preprocess function for CameraLoader.
    """
    image = resize_fn(image)    # (h, w, c) -> (h, w, c)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # (h, w, c) -> (c, h, w)
    return image    # (h, w, c)


def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))   # (left, top, right, bottom)

class FallDetector:
    isOpened = False
    def __init__(self):
        self.stop = False
        self.cam_source = "0"  # 0: webcam, 1: usb cam, 2: ip cam, 3: rtsp cam.
        self.detection_input_size = 384  # 320, 416, 512, 608.
        self.pose_input_size = "224x160"  # 256x192, 384x288, 512x384, 640x480, 736x736, 800x608.
        self.pose_backbone = "resnet50" # resnet50, resnet101, resnet152.
        self.show_detected = False  # show all bounding box from detection.
        self.show_skeleton = True  # show skeleton pose.
        self.save_out = "fall.mp4v" # Save display to video file.
        self.device = "cuda"  # cpu or cuda 
        # DETECTION MODEL.
        inp_dets = self.detection_input_size
        self.detect_model = TinyYOLOv3_onecls(inp_dets, device=self.device)

        # POSE MODEL.
        inp_pose = self.pose_input_size.split('x') # (h, w) for SPPE FastPose model.
        inp_pose = (int(inp_pose[0]), int(inp_pose[1]))     # (h, w) for SPPE FastPose model.
        self.pose_model = SPPE_FastPose(self.pose_backbone, inp_pose[0], inp_pose[1], device=self.device) # (backbone, h, w, device)

        # the size of input image for pose model must be divisible by 32.
        # so we need to resize and padding image to square.

        # Tracker.
        max_age = 30
        self.tracker = Tracker(max_age=max_age, n_init=3)    # (max_age, n_init)

        # Actions Estimate.
        self.action_model = TSSTG()  # (n_classes, n_frames, n_joints, n_features)

        resize_fn = ResizePadding(inp_dets, inp_dets) # Resize and padding image to square.

        if type(self.cam_source) is str and os.path.isfile(self.cam_source):  # Check if video file path.
         # Use loader thread with Q for video file.
            self.cam = CamLoader_Q(self.cam_source, queue_size=1000, preprocess=lambda x: preproc(x, resize_fn)).start()  # (w, h) for video file.
        else:
            # Use normal thread loader for webcam.
            self.cam = CamLoader(int(self.cam_source) if self.cam_source.isdigit() else self.cam_source,    # Camera index or video file path. 
                        preprocess=lambda x: preproc(x, resize_fn)).start() # (w, h)


        #frame_size = self.cam.frame_size
        #scf = torch.min(inp_size / torch.FloatTensor([frame_size]), 1)[0]

        self.outvid = False
        if self.save_out != '':
            self.outvid = True
            codec = cv2.VideoWriter_fourcc(*'mp4v') # 'x264' doesn't work
            self.writer = cv2.VideoWriter(self.save_out, codec, 30, (inp_dets * 2, inp_dets * 2)) # (w, h)

        self.fps_time = 0
        self.f = 0
        
    def get_frame(self):
        while self.cam.grabbed():    # Loop until stop.
            self.f += 1
            
            model_file = os.path.join(os.path.dirname(current_file), 'modules', 'model.h5') # load the model file
            model.load_weights(model_file) # load the weights
            cv2.ocl.setUseOpenCL(False) # to avoid error
            emotion_dict = {0: "Angry", 1: "Disgusted", 
                2: "Fearful", 3: "Happy",
                4: "Neutral", 5: "Sad",
                6: "Surprised"} # dictionary of emotions
            
            facecasc_file = os.path.join(os.path.dirname(current_file), 'modules', 'haarcascade_frontalface_default.xml') # load the cascade file
            facecasc = cv2.CascadeClassifier(facecasc_file) # load the cascade
            
            frame = self.cam.getitem()   # Get frame from camera loader.
            image = frame.copy()    # Copy frame for display.
            
            # Detect humans bbox in the frame with detector model.
            detected = self.detect_model.detect(frame, need_resize=False, expand_bb=10)

            # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
            self.tracker.predict()
            # Merge two source of predicted bbox together.
            for track in self.tracker.tracks:
                det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
                detected = torch.cat([detected, det], dim=0) if detected is not None else det

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
          
            detections = []  # List of Detections object for tracking.
            if detected is not None:
                #detected = non_max_suppression(detected[None, :], 0.45, 0.2)[0]
                # Predict skeleton pose of each bboxs.
                poses = self.pose_model.predict(frame, detected[:, 0:4], detected[:, 4]) # (x1, y1, x2, y2, score)

                # Create Detections object.
                detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),  # (x1, y1, x2, y2)
                                        np.concatenate((ps['keypoints'].numpy(),    # (x, y)
                                                        ps['kp_score'].numpy()), axis=1),   # (x, y, score)
                                        ps['kp_score'].mean().numpy()) for ps in poses] # score

                # VISUALIZE.
                if self.show_detected:  # Show all detected bbox.
                    for bb in detected[:, 0:5]: # (x1, y1, x2, y2, score)
                        frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 1)    # Draw bbox.

            # Update tracks by matching each track information of current and previous frame or
            # create a new track if no matched.
            self.tracker.update(detections)  # Update tracks.

            # Predict Actions of each track.
            for i, track in enumerate(self.tracker.tracks):
                if not track.is_confirmed():    # Skip unconfirmed tracks.
                    continue    # Skip unconfirmed tracks.

                track_id = track.track_id   # Get the ID for this track.
                bbox = track.to_tlbr().astype(int)  # Get the corrected/predicted bbox
                center = track.get_center().astype(int) # Get the center point

                action = 'Pending..'    # Default action when not enough data.
                clr = (0, 255, 0)   # Default color is green.
                if not os.path.exists('extra_data.csv'): # if file does not exist write header
                        with open('extra_data.csv', 'w', newline='') as file:
                            csvWriter = csv.writer(file)
                            csvWriter.writerow(["Frame", "Action", "Probability", "Action Prob Threshold", "Time Stamp"])      
                # Use 30 frames time-steps to prediction.
                if len(track.keypoints_list) == 30: # Enough data for prediction.
                    pts = np.array(track.keypoints_list, dtype=np.float32)
                    out = self.action_model.predict(pts, frame.shape[:2])    # Predict action.
                    action_name = self.action_model.class_names[out[0].argmax()] # Get action name.
                    action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)  # Set action text.
                    if action_name == 'Fall Down':  # Set color to red if action is fall down.
                        clr = (255, 0, 0)   # Set color to red.
                        if not os.path.exists('fall_data.csv'): # if file does not exist write header
                            with open('fall_data.csv', 'w', newline='') as file:
                                csvWriter = csv.writer(file)
                                csvWriter.writerow(["Frame", "Fall", "Fall Prob", "Fall Prob Threshold", "Time Stamp"])      
                        else:
                            with open('fall_data.csv', 'a', newline='') as file: # a for append
                                csvWriter = csv.writer(file)
                                csvWriter.writerow([self.f, action_name, out[0].max(), 0.5, datetime.datetime.now()])
                    elif action_name == 'Lying Down':   # Set color to yellow if action is lying down.
                        clr = (255, 200, 0)
                        with open('extra_data.csv', 'a', newline='') as file:
                            csvWriter = csv.writer(file)
                            csvWriter.writerow([self.f, action_name, out[0].max(), 0.5, datetime.datetime.now()])
                    elif action_name == 'Sitting':
                        clr = (0, 0, 255) #color for sitting down is blue
                        with open('extra_data.csv', 'a', newline='') as file:
                            csvWriter = csv.writer(file)
                            csvWriter.writerow([self.f, action_name, out[0].max(), 0.5, datetime.datetime.now()])
                    elif action_name == 'Stand Up':
                        clr = (0, 255, 0) #color for standing up is green which is default
                        with open('extra_data.csv', 'a', newline='') as file:
                            csvWriter = csv.writer(file)
                            csvWriter.writerow([self.f, action_name, out[0].max(), 0.5, datetime.datetime.now()])
                    elif action_name == 'Walking':
                        clr = (255, 0, 255) #color for walking is purple
                        with open('extra_data.csv', 'a', newline='') as file:
                            csvWriter = csv.writer(file)
                            csvWriter.writerow([self.f, action_name, out[0].max(), 0.5, datetime.datetime.now()])
                    elif action_name == "Standing":
                        clr = (0, 255, 255) #color for standing is yellow
                        with open('extra_data.csv', 'a', newline='') as file:
                            csvWriter = csv.writer(file)
                            csvWriter.writerow([self.f, action_name, out[0].max(), 0.5, datetime.datetime.now()])

                # VISUALIZE.
                if track.time_since_update == 0:    # Draw only the last bbox of the track.
                    if self.show_skeleton:  # Show skeleton.
                        frame = draw_single(frame, track.keypoints_list[-1])    # Draw skeleton.
                    frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)    # Draw bbox.
                    frame = cv2.putText(frame, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX,     # Draw track id.
                                        0.4, (255, 0, 0), 2)    # Draw track id.
                    frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX,   # Draw action.
                                        0.4, clr, 1)    # Draw action.

            # Show Frame.
            frame = cv2.resize(frame, (0, 0), fx=2., fy=2.) # Resize frame for display.
            frame = cv2.putText(frame, '%d, FPS: %f' % (self.f, 1.0 / (time.time() - self.fps_time)),     # Draw FPS.
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)    # Draw FPS.
            frame = frame[:, :, ::-1]   # Convert to RGB for display.
            self.fps_time = time.time()  # Record time for calculating FPS.

            if self.outvid:  # Write video.
                self.writer.write(frame) # Write frame to video.

            ret, jpeg = cv2.imencode('.jpg', frame)  # Encode frame to jpg.
            if isOpened:
                return jpeg.tobytes()   # Return frame.
            
            # cv2.imshow('frame', frame)  # Show frame.

            if self.stop:
                self.close()
                
    def close(self):
    # Clear resource.
        self.cam.stop()  # Stop camera loader.
        if self.outvid:  # Release video writer.
            self.writer.release()    # Release video writer.
        self.stop = True    # Stop thread.
        isOpened = False
      