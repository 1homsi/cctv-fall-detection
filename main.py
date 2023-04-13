import logging
from tkinter import Tk, messagebox
import eel
import base64

import os
import sys
sys.path.append(os.path.abspath(''))


from emotion.emotion import *
from fall.fall import *
# from FallEmotion.mix import *

# Disable the ability to resize the window and set the window size to 800x600
window_size = (800, 600)

def show_error(title, msg):
    root = Tk() # create a tkinter window
    root.withdraw()  # hide main window
    messagebox.showerror(title, msg) # show error message
    root.destroy() # destroy the tkinter window

def gen(camera):
    while camera.isOpened:
        frame = camera.get_frame()
        if frame is not None:
            yield frame
        else:
            break

@eel.expose
def video_feed():
    x = VideoCamera()
    y = gen(x)
    for each in y:
        blob = base64.b64encode(each)
        blob = blob.decode("utf-8")
        eel.updateImageSrc(blob)()
    x.close_camera()

@eel.expose
def train(iterations):
    if Train_Model(int(iterations)): # train the model
        return "Model Was trained Successfully"
    
@eel.expose
def Close():
    VideoCamera().close_camera() #close the camera
    
@eel.expose
def detectFallFeed(source):
    if FallDetector.isOpened:
        x = FallDetector(source) # create a FallDetector object
        y = gen(x) # create a generator object
        for each in y: # iterate through the generator object
            blob = base64.b64encode(each) # encode the frame
            blob = blob.decode("utf-8") # decode the frame
            eel.updateImageSrc(blob)() # update the image source
    else:
        print("Fall Detector is not opened")
        
@eel.expose
def CloseDetector():
    FallDetector().close()   # close the camera
        
if __name__ == "__main__":
    # Start the server 
    try:
        eel.init('client') # path to project folder 
        eel.start('index.html', size=window_size)
    except Exception as e: 
        err_msg = 'Could not launch a local server' # error message
        logging.error('{}\n{}'.format(err_msg, e.args))
        show_error(title='Failed to initialise server',
                   msg=err_msg) #use tkinter to show error message
        sys.exit()
