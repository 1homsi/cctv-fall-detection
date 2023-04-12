import logging
from tkinter import Tk, messagebox
import eel
import base64

import os
import sys
sys.path.append(os.path.abspath(''))


from emotion.emotion import *
from fall.fall import *

def show_error(title, msg):
    root = Tk() # create a tkinter window
    root.withdraw()  # hide main window
    messagebox.showerror(title, msg) # show error message
    root.destroy() # destroy the tkinter window

def gen(camera):
    while True:
        frame = camera.get_frame() # get frame from camera
        """ Yield is a keyword in Python that is used to return 
        from a function without destroying the states of 
        its local variable """
        yield frame # yield the frame

@eel.expose
def video_feed():
    if VideoCamera.isOpened: 
        x = VideoCamera() # create a VideoCamera object
        y = gen(x) # create a generator object
        for each in y: # iterate through the generator object
            blob = base64.b64encode(each) # encode the frame
            blob = blob.decode("utf-8") # decode the frame
            eel.updateImageSrc(blob)() # update the image source
    else:
        print("Camera is not opened")

@eel.expose
def train(iterations):
    if Train_Model(int(iterations)): # train the model
        return "Model Was trained Successfully"
    
@eel.expose
def Close():
    VideoCamera().close_camera() #close the camera
    print("Camera Closed")
    
@eel.expose
def detectFallFeed():
    if FallDetector.isOpened:
        x = FallDetector() # create a FallDetector object
        y = gen(x) # create a generator object
        for each in y: # iterate through the generator object
            blob = base64.b64encode(each) # encode the frame
            blob = blob.decode("utf-8") # decode the frame
            eel.updateImageSrc(blob)() # update the image source
    else:
        print("Fall Detector is not opened")
        
if __name__ == "__main__":
    # Start the server 
    try:
        eel.init('client') # path to project folder 
        eel.start('index.html') # start the web app with the main file index.html                
    except Exception as e: 
        err_msg = 'Could not launch a local server' # error message
        logging.error('{}\n{}'.format(err_msg, e.args))
        show_error(title='Failed to initialise server',
                   msg=err_msg) #use tkinter to show error message
        sys.exit()
