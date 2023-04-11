import os
import cv2
import time
import torch
import argparse
import numpy as np 
import csv
import datetime

from Detection.Utils import ResizePadding
from CameraLoader import CamLoader, CamLoader_Q
from DetectorLoader import TinyYOLOv3_onecls

from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single

from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG

#source = '../Data/test_video/test7.mp4'
#source = '../Data/falldata/Home/Videos/video (2).avi'  # hard detect
# source = '../Data/falldata/Home/Videos/video (1).avi'
#source = 2


def preproc(image):
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


if __name__ == '__main__':
    par = argparse.ArgumentParser(description='Human Fall Detection Demo.')
    par.add_argument('-C', '--camera', default="0",  # required=True,  # default=2,
                        help='Source of camera or video file path.')    # 0: webcam, 1: usb cam, 2: ip cam, 3: rtsp cam.
    par.add_argument('--detection_input_size', type=int, default=384,
                        help='Size of input in detection model in square must be divisible by 32 (int).') # 320, 416, 512, 608.
    par.add_argument('--pose_input_size', type=str, default='224x160',
                        help='Size of input in pose model must be divisible by 32 (h, w)')  # 256x192, 384x288, 512x384, 640x480, 736x736, 800x608.
    par.add_argument('--pose_backbone', type=str, default='resnet50',
                        help='Backbone model for SPPE FastPose model.') # resnet50, resnet101, resnet152.
    par.add_argument('--show_detected', default=False, action='store_true',
                        help='Show all bounding box from detection.')   # show all bounding box from detection.
    par.add_argument('--show_skeleton', default=True, action='store_true',
                        help='Show skeleton pose.')  # show skeleton pose.
    par.add_argument('--save_out', type=str, default='fall.mp4v',
                        help='Save display to video file.') # Save display to video file.
    par.add_argument('--device', type=str, default='cuda',
                        help='Device to run model on cpu or cuda.') # cpu or cuda.
    args = par.parse_args() # Parse arguments.

    device = args.device    # Device to run model on cpu or cuda.
        
    # DETECTION MODEL.
    inp_dets = args.detection_input_size
    detect_model = TinyYOLOv3_onecls(inp_dets, device=device)

    # POSE MODEL.
    inp_pose = args.pose_input_size.split('x') # (h, w) for SPPE FastPose model.
    inp_pose = (int(inp_pose[0]), int(inp_pose[1]))     # (h, w) for SPPE FastPose model.
    pose_model = SPPE_FastPose(args.pose_backbone, inp_pose[0], inp_pose[1], device=device) # (backbone, h, w, device)
    
    # the size of input image for pose model must be divisible by 32.
    # so we need to resize and padding image to square.
    
    # Tracker.
    max_age = 30
    tracker = Tracker(max_age=max_age, n_init=3)    # (max_age, n_init)

    # Actions Estimate.
    action_model = TSSTG()  # (n_classes, n_frames, n_joints, n_features)

    resize_fn = ResizePadding(inp_dets, inp_dets) # Resize and padding image to square.

    cam_source = args.camera    # Camera index or video file path.
    if type(cam_source) is str and os.path.isfile(cam_source):  # Check if video file path.
        # Use loader thread with Q for video file.
        cam = CamLoader_Q(cam_source, queue_size=1000, preprocess=preproc).start()  # (w, h) for video file.
    else:
        # Use normal thread loader for webcam.
        cam = CamLoader(int(cam_source) if cam_source.isdigit() else cam_source,    # Camera index or video file path. 
                        preprocess=preproc).start() # (w, h)

    #frame_size = cam.frame_size
    #scf = torch.min(inp_size / torch.FloatTensor([frame_size]), 1)[0]

    outvid = False
    if args.save_out != '':
        outvid = True
        codec = cv2.VideoWriter_fourcc(*'mp4v') # 'x264' doesn't work
        writer = cv2.VideoWriter(args.save_out, codec, 30, (inp_dets * 2, inp_dets * 2)) # (w, h)

    fps_time = 0
    f = 0
    while cam.grabbed():    # Loop until stop.
        f += 1
        frame = cam.getitem()   # Get frame from camera loader.
        image = frame.copy()    # Copy frame for display.

        # Detect humans bbox in the frame with detector model.
        detected = detect_model.detect(frame, need_resize=False, expand_bb=10)

        # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
        tracker.predict()
        # Merge two source of predicted bbox together.
        for track in tracker.tracks:
            det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
            detected = torch.cat([detected, det], dim=0) if detected is not None else det

        detections = []  # List of Detections object for tracking.
        if detected is not None:
            #detected = non_max_suppression(detected[None, :], 0.45, 0.2)[0]
            # Predict skeleton pose of each bboxs.
            poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4]) # (x1, y1, x2, y2, score)

            # Create Detections object.
            detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),  # (x1, y1, x2, y2)
                                    np.concatenate((ps['keypoints'].numpy(),    # (x, y)
                                                    ps['kp_score'].numpy()), axis=1),   # (x, y, score)
                                    ps['kp_score'].mean().numpy()) for ps in poses] # score

            # VISUALIZE.
            if args.show_detected:  # Show all detected bbox.
                for bb in detected[:, 0:5]: # (x1, y1, x2, y2, score)
                    frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 1)    # Draw bbox.

        # Update tracks by matching each track information of current and previous frame or
        # create a new track if no matched.
        tracker.update(detections)  # Update tracks.

        # Predict Actions of each track.
        for i, track in enumerate(tracker.tracks):
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
                out = action_model.predict(pts, frame.shape[:2])    # Predict action.
                action_name = action_model.class_names[out[0].argmax()] # Get action name.
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
                            csvWriter.writerow([f, action_name, out[0].max(), 0.5, datetime.datetime.now()])
                elif action_name == 'Lying Down':   # Set color to yellow if action is lying down.
                    clr = (255, 200, 0)
                    with open('extra_data.csv', 'a', newline='') as file:
                        csvWriter = csv.writer(file)
                        csvWriter.writerow([f, action_name, out[0].max(), 0.5, datetime.datetime.now()])
                elif action_name == 'Sitting':
                    clr = (0, 0, 255) #color for sitting down is blue
                    with open('extra_data.csv', 'a', newline='') as file:
                        csvWriter = csv.writer(file)
                        csvWriter.writerow([f, action_name, out[0].max(), 0.5, datetime.datetime.now()])
                elif action_name == 'Stand Up':
                    clr = (0, 255, 0) #color for standing up is green which is default
                    with open('extra_data.csv', 'a', newline='') as file:
                        csvWriter = csv.writer(file)
                        csvWriter.writerow([f, action_name, out[0].max(), 0.5, datetime.datetime.now()])
                elif action_name == 'Walking':
                    clr = (255, 0, 255) #color for walking is purple
                    with open('extra_data.csv', 'a', newline='') as file:
                        csvWriter = csv.writer(file)
                        csvWriter.writerow([f, action_name, out[0].max(), 0.5, datetime.datetime.now()])
                elif action_name == "Standing":
                    clr = (0, 255, 255) #color for standing is yellow
                    with open('extra_data.csv', 'a', newline='') as file:
                        csvWriter = csv.writer(file)
                        csvWriter.writerow([f, action_name, out[0].max(), 0.5, datetime.datetime.now()])
                       
            # VISUALIZE.
            if track.time_since_update == 0:    # Draw only the last bbox of the track.
                if args.show_skeleton:  # Show skeleton.
                    frame = draw_single(frame, track.keypoints_list[-1])    # Draw skeleton.
                frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)    # Draw bbox.
                frame = cv2.putText(frame, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX,     # Draw track id.
                                    0.4, (255, 0, 0), 2)    # Draw track id.
                frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX,   # Draw action.
                                    0.4, clr, 1)    # Draw action.

        # Show Frame.
        frame = cv2.resize(frame, (0, 0), fx=2., fy=2.) # Resize frame for display.
        frame = cv2.putText(frame, '%d, FPS: %f' % (f, 1.0 / (time.time() - fps_time)),     # Draw FPS.
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)    # Draw FPS.
        frame = frame[:, :, ::-1]   # Convert to RGB for display.
        fps_time = time.time()  # Record time for calculating FPS.

        if outvid:  # Write video.
            writer.write(frame) # Write frame to video.

        cv2.imshow('frame', frame)  # Show frame.
                
        if cv2.waitKey(1) & 0xFF == ord('q'):   # Press q to
            break

    # Clear resource.
    cam.stop()  # Stop camera loader.
    if outvid:  # Release video writer.
        writer.release()    # Release video writer.
    cv2.destroyAllWindows() # Close all windows.
